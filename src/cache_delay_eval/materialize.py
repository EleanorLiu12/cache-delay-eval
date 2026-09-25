"""Materialize synthetic session hashes as reproducible, opaque token prompts.

This preserves block-prefix identity, not semantic text or generated answers.
Only traces produced by our session generator are supported.
"""
from dataclasses import replace
from .session_gen import _stream_block_hashes
from .trace import TraceHeader, TraceRequest


def materialize(header: TraceHeader, requests: list[TraceRequest],
                alphabet: list[int], model: str):
    size = header.block_size
    if header.source != "synthetic-agent-sessions" or not size:
        raise ValueError("only synthetic-agent-sessions with a block size are supported")
    if len(set(alphabet)) != len(alphabet) or len(alphabet) < 2:
        raise ValueError("alphabet must contain distinct token IDs")
    if any(type(t) is not int or t < 0 for t in alphabet):
        raise ValueError("invalid token ID")
    if len(alphabet) ** size < 2**64:
        raise ValueError("alphabet/block size cannot encode a 64-bit identity injectively")
    prefix = header.metadata["P"]
    if prefix % size:
        raise ValueError("pilot requires a block-aligned shared prefix")
    encoded = {}

    def block_tokens(identity):
        if identity not in encoded:
            digits = []
            value = identity
            for _ in range(size):
                value, digit = divmod(value, len(alphabet))
                digits.append(alphabet[digit])
            encoded[identity] = tuple(digits)
        return encoded[identity]

    output = []
    for request in requests:
        session = int(request.session_id.removeprefix("session-"))
        group = int(request.prefix_group.removeprefix("prefix-"))
        count = (request.prompt_tokens + size - 1) // size
        hashes = _stream_block_hashes(count, prefix // size, group, session)
        if tuple(hashes[:request.prompt_tokens // size]) != request.block_hashes:
            raise ValueError(f"{request.request_id}: hashes disagree with synthetic session metadata")
        tokens = tuple(t for h in hashes for t in block_tokens(h))[:request.prompt_tokens]
        output.append(replace(request, block_hashes=None, token_ids=tokens,
                              metadata={**request.metadata, "materialization": "hash-base-alphabet-v1"}))
    return replace(header, model_id=model, tokenizer_id=model,
                   metadata={**header.metadata, "token_alphabet": alphabet,
                             "materialization": "hash-base-alphabet-v1",
                             "semantics": "opaque synthetic tokens; generated outputs not fed into later prompts"}), output
