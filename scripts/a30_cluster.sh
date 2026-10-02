#!/usr/bin/env bash
# Four isolated vLLM instances on one A30 via Multi-Instance GPU (4 x 1g.6gb).
#   mig-on   enable MIG and create four 1g.6gb instances (needs sudo)
#   launch   start one vLLM server per MIG device (ports 8001-8004, events 5601-5604)
#   shared   fallback: four servers on the whole GPU, no compute isolation
#   status   health of every server
#   stop     stop all servers
#   mig-off  destroy MIG instances and disable MIG
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
REVISION=${REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}
MAX_LEN=${MAX_LEN:-9216}
UTIL=${UTIL:-0.85}
LOG_DIR=${LOG_DIR:-logs}
N=4

serve() {  # serve <index> <device> <gpu-memory-utilization>
  local i=$1 port=$((8001 + $1)) events=$((5601 + $1))
  CUDA_VISIBLE_DEVICES=$2 VLLM_SERVER_DEV_MODE=1 nohup vllm serve "$MODEL" \
    --revision "$REVISION" --port "$port" --seed 0 \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$3" \
    --enable-prefix-caching --enable-prompt-tokens-details \
    --kv-events-config "{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\", \"endpoint\": \"tcp://*:$events\"}" \
    > "$LOG_DIR/vllm-$i.log" 2>&1 &
  echo "instance $i: device $2, http://127.0.0.1:$port, tcp://127.0.0.1:$events"
}

case ${1:-} in
  mig-on)
    sudo nvidia-smi -i 0 -mig 1
    nvidia-smi -i 0 --query-gpu=mig.mode.current,mig.mode.pending --format=csv
    # If current is still Disabled: stop users of the GPU, then
    #   sudo systemctl stop nvidia-persistenced; sudo nvidia-smi --gpu-reset -i 0
    # and rerun mig-on. If the reset is refused, reboot the node.
    sudo nvidia-smi mig -i 0 -cgi 1g.6gb,1g.6gb,1g.6gb,1g.6gb -C
    nvidia-smi -L
    ;;
  launch)
    mkdir -p "$LOG_DIR"
    mapfile -t uuids < <(nvidia-smi -L | grep -o 'MIG-[0-9a-f-]*')
    [[ ${#uuids[@]} -eq $N ]] || { echo "expected $N MIG devices, found ${#uuids[@]}"; exit 1; }
    for i in $(seq 0 $((N - 1))); do serve "$i" "${uuids[$i]}" "$UTIL"; done
    ;;
  shared)
    mkdir -p "$LOG_DIR"
    for i in $(seq 0 $((N - 1))); do serve "$i" 0 0.2; sleep 60; done  # stagger memory profiling
    ;;
  status)
    for i in $(seq 0 $((N - 1))); do
      printf 'instance %d: ' "$i"
      curl -s -o /dev/null -w '%{http_code}\n' "http://127.0.0.1:$((8001 + i))/health" || echo down
    done
    ;;
  stop)
    pkill -f "vllm serve $MODEL" || true
    ;;
  mig-off)
    sudo nvidia-smi mig -i 0 -dci && sudo nvidia-smi mig -i 0 -dgi && sudo nvidia-smi -i 0 -mig 0
    ;;
  *)
    echo "usage: $0 {mig-on|launch|shared|status|stop|mig-off}"; exit 2 ;;
esac
