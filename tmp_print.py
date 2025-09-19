import pathlib
text = pathlib.Path('test/core/test_config_validator.jl').read_text(encoding='utf-8').replace('\r\n','\n')
block_start = text.index('# ascii params mapping: s -> σ')
block_end = text.index('# β out of range', block_start)
print(text[block_start:block_end].encode('unicode_escape').decode())
