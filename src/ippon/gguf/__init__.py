"""GGUF builds of the supported models (llama.cpp's format), in Q8_0 as their creators publish them.

    Decider.load("LiquidAI/d1-2B-GGUF:Q8_0")  # a build ippon knows (verified.py, builds.py), from the Hub
    Decider.load("~/models/d1-2B-Q8_0.gguf")  # the same file, downloaded already

A build is a model's source (source.py): the creator's small files (config, tokenizer) from the pinned base
repository, beside the GGUF file, whose metadata is checked against them (`source.bound`). Its header (header.py) is
read without MLX; its tensors are read straight into MLX (reader.py) and renamed to the creator's checkpoint's, with
llama.cpp's transforms undone (names.py, tensors.py), so each adapter reads it as it reads that checkpoint. locate.py
downloads a build named by its spec.
"""
