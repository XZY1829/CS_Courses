"""Extract text from PPT (old format) files using win32com."""
import os
from pathlib import Path

import win32com.client
import pythoncom


def cpp_directory() -> Path:
    return Path(__file__).resolve().parent


def main():
    pythoncom.CoInitialize()
    ppt_app = win32com.client.Dispatch('PowerPoint.Application')

    cpp_dir = cpp_directory()
    output_dir = os.path.join(cpp_dir, "_extracted")
    os.makedirs(output_dir, exist_ok=True)
    
    ppt_files = ['9-面向对象的IO.ppt', '12 函数式程序设计.ppt', '13事件驱动的程序设计.ppt']
    
    for filename in ppt_files:
        filepath = os.path.join(cpp_dir, filename)
        base = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, base + ".txt")
        
        try:
            presentation = ppt_app.Presentations.Open(filepath, ReadOnly=True, WithWindow=False)
            slide_count = presentation.Slides.Count
            lines = []
            for i in range(1, slide_count + 1):
                slide = presentation.Slides(i)
                sep = "=" * 60
                lines.append("")
                lines.append(sep)
                lines.append(f"=== Slide {i} ===")
                lines.append(sep)
                lines.append("")
                for j in range(1, slide.Shapes.Count + 1):
                    shape = slide.Shapes(j)
                    if shape.HasTextFrame:
                        text = shape.TextFrame.TextRange.Text.strip()
                        if text:
                            lines.append(text)
            presentation.Close()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"OK: {filename} -> {slide_count} slides")
        except Exception as e:
            print(f"ERROR: {filename}: {e}")
    
    ppt_app.Quit()
    pythoncom.CoUninitialize()

if __name__ == '__main__':
    main()
