"""Exercise real selected user scripts and pinned tokenizer against a CPU mock.

All replies and cache counters come from the explicitly named mock, never a
model. Timings test client dependencies; they are not serving measurements.
"""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from aiohttp import web

from cache_delay_eval.chat_replay import QwenPromptBuilder, ReplayConfig, load_scripts, replay_sessions


async def execute(plan_dir, output):
    output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((plan_dir/"plan.json").read_text())
    builder=QwenPromptBuilder("data/wildchat-audit-2026-09-28/tokenizer", "data/wildchat-audit-2026-09-28/manifest.json")
    request_evidence=[]
    active=0
    peak=0
    async def serve(request):
        nonlocal active,peak
        body=await request.json()
        active+=1
        peak=max(peak,active)
        start=asyncio.get_running_loop().time()
        text=f"CPU MOCK reply for {body['request_id'].split(':')[-1]}. Retained generated content."
        ids=builder.tokenizer.encode(text,add_special_tokens=False)+[builder.tokenizer.eos_token_id]
        response=web.StreamResponse(headers={"Content-Type":"text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(.004)
        events=[dict(id="cmpl-"+body["request_id"],choices=[dict(index=0,text="",token_ids=ids[:1],finish_reason=None)]),
                dict(id="cmpl-"+body["request_id"],choices=[dict(index=0,text=text,token_ids=ids[1:],finish_reason="stop")]),
                dict(id="cmpl-"+body["request_id"],choices=[],usage=dict(prompt_tokens=len(body["prompt"]),completion_tokens=len(ids),prompt_tokens_details=dict(cached_tokens=0)))]
        for index,event in enumerate(events):
            if index==1:
                await asyncio.sleep(.004)
            data=("data: "+json.dumps(event)+"\n\n").encode()
            await response.write(data[:9])
            await response.write(data[9:])
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        request_evidence.append(dict(request_id=body["request_id"],header_request_id=request.headers.get("X-Request-Id"),
            server_received_monotonic_s=start,server_finished_monotonic_s=asyncio.get_running_loop().time(),
            prompt_token_ids=body["prompt"],returned_token_ids=ids,returned_content=text,
            cache_counter_source="constant zero from CPU mock; no cache modeled"))
        active-=1
        return response
    app=web.Application(client_max_size=10*1024*1024)
    app.router.add_post("/v1/completions",serve)
    runner=web.AppRunner(app)
    await runner.setup()
    site=web.TCPSite(runner,"127.0.0.1",0)
    await site.start()
    base=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    results={}
    try:
        for label in ("smoke","experiment","calibration"):
            scripts=load_scripts(plan["groups"][label]["source_scripts"],plan_dir/f"{label}-mock-schedule.json")
            config=ReplayConfig(run_id=f"{label}-cpu",max_tokens=512,think_delay_s=.003,timeout_s=30,max_dispatch_lag_ms=5000)
            summary=await replay_sessions(scripts,builder,base,output/label,config,
                 run_meta=dict(server="local CPU mock",model_inference=False,weights_downloaded=False,
                               think_time_assumption="3 milliseconds for fast correctness validation; GPU candidate remains 5 seconds"))
            if not summary["valid"]:
                raise AssertionError(f"CPU mock replay failed: {label} {summary}")
            rows=[json.loads(l) for l in (output/label/"turns.jsonl").open()]
            by_id={r["request_id"]:r for r in rows}
            observed={r["request_id"]:r for r in request_evidence}
            for row in rows:
                server=observed[row["request_id"]]
                assert row["output_token_ids"]==server["returned_token_ids"]
                assert row["assistant_content"]==server["returned_content"]
                assert server["header_request_id"]==row["request_id"]
                assert row["first_token_s"]<row["first_text_s"]
                if row["parent_request_id"]:
                    parent=by_id[row["parent_request_id"]]
                    assert row["history_messages"][-2]==dict(role="assistant",content=parent["assistant_content"])
                    assert server["server_received_monotonic_s"]>=observed[parent["request_id"]]["server_finished_monotonic_s"]+.003
            results[label]=dict(summary=summary,independent_server_checks=len(rows),
                                all_generated_content_reused=True,no_source_assistant_substitution=True)
    finally:
        await runner.cleanup()
    (output/"mock-server-requests.jsonl").write_text("".join(json.dumps(r)+"\n" for r in request_evidence))
    result=dict(cpu_only=True,model_inference=False,mock_server_peak_concurrency=peak,
                scripts_total=sum(r["summary"]["planned_requests"] for r in results.values()),groups=results,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                interpretation="Correctness validation with fabricated mock replies and cache counters; no latency/cache-hit/performance result")
    (output/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(turns=result["scripts_total"],peak_concurrency=peak,groups={k:v["summary"]["valid"] for k,v in results.items()})))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    asyncio.run(execute(args.plan_dir,args.output_dir))


if __name__=="__main__":
    main()
