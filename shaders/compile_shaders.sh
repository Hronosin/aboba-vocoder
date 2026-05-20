#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Compile GLSL shaders to SPIR-V, then embed each as a C header containing
# a static uint32_t array. This means the runtime needs no on-disk shader
# files — the SPIR-V is baked into the binary.
#
# Run from the project root.
set -euo pipefail

SHADER_DIR="shaders"
OUT_DIR="src/vulkan_shaders_gen"

mkdir -p "$OUT_DIR"

for shader in fft_stage.comp r2c_pack.comp c2r_finish.comp bitrev_permute.comp; do
    name="${shader%.comp}"
    src="$SHADER_DIR/$shader"
    spv="$OUT_DIR/$name.spv"
    hdr="$OUT_DIR/$name.h"

    echo "Compiling $shader..."
    glslangValidator -V -S comp -o "$spv" "$src"

    # Generate a C header from the SPIR-V binary
    python3 -c "
import sys
data = open('$spv', 'rb').read()
words = []
for i in range(0, len(data), 4):
    words.append(int.from_bytes(data[i:i+4], 'little'))
out = open('$hdr', 'w')
out.write('// Auto-generated. DO NOT EDIT.\\n')
out.write('// Compiled from $shader\\n')
out.write('#pragma once\\n#include <cstdint>\\n')
out.write('namespace aboba { namespace vk_shaders {\\n')
out.write(f'inline constexpr std::uint32_t ${name}_spv[] = {{\\n')
for i, w in enumerate(words):
    if i % 8 == 0: out.write('  ')
    out.write(f'0x{w:08x}u,')
    if i % 8 == 7: out.write('\\n')
    else:          out.write(' ')
if len(words) % 8 != 0: out.write('\\n')
out.write('};\\n')
out.write(f'inline constexpr std::size_t ${name}_spv_size_bytes = {len(data)};\\n')
out.write('}}  // namespace\\n')
out.close()
print(f'  -> $hdr ({len(words)} words = {len(data)} bytes)')
"
done

echo "Done."
