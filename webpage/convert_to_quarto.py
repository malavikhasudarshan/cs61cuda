#!/usr/bin/env python3
"""
Convert README.md to Quarto format with table of contents.
"""

import re
import sys

def convert_markdown_to_quarto(readme_path, output_path):
    """Convert README.md to Quarto .qmd format."""
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove extra asterisks from headings (clean up markdown)
    content = re.sub(r'^## \*\*(.+?)\*\*$', r'## \1', content, flags=re.MULTILINE)
    content = re.sub(r'^### \*\*(.+?)\*\*$', r'### \1', content, flags=re.MULTILINE)
    content = re.sub(r'^# \*\*(.+?)\*\*$', r'# \1', content, flags=re.MULTILINE)
    
    # Clean up escaped characters
    content = content.replace('\\!\\!', '!!')
    content = content.replace('\\‑', '-')
    
    # Create Quarto document with YAML frontmatter
    quarto_content = '''---
title: "CS61Cuda Project: Matmul & CUDA Fundamentals"
format:
  html:
    toc: true
    toc-depth: 3
    toc-location: left
    toc-title: "Contents"
    number-sections: true
    code-fold: true
    code-tools: false
    theme: cosmo
    css: styles.css
    fig-width: 8
    fig-height: 6
---

''' + content
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(quarto_content)
    
    print(f"✓ Converted {readme_path} to {output_path}")
    print(f"✓ Quarto document created with table of contents sidebar")

if __name__ == '__main__':
    readme_path = '../README.md'
    output_path = 'webpage.qmd'
    
    if len(sys.argv) > 1:
        readme_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    convert_markdown_to_quarto(readme_path, output_path)

