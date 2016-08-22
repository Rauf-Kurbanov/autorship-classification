#! /bin/bash

PATH_TO_DIR="$1"
OUT_FILE="$2"

touch "$OUT_FILE"
> "$OUT_FILE"

tmpfile=$(mktemp /tmp/XXXXX.txt)

# fb2 files
DECODER_PATH="/home/rauf/Soft/fb2-2-txt/FB2_2_txt.xsl"
find "$PATH_TO_DIR" -name "*.fb2" -print0 |
    while IFS= read -r -d $'\0' line; do
        echo processing "$line"
        xsltproc "$DECODER_PATH" "$line" >> "$OUT_FILE"
    done

# epub files
find "$PATH_TO_DIR" -name "*.epub" -print0 |
    while IFS= read -r -d $'\0' line; do
        echo processing "$line"
        ebook-convert "$line" "$tmpfile"

        cat "$tmpfile" >> "$OUT_FILE"
    done

# pdf files
find "$PATH_TO_DIR" -name "*.pdf" -print0 |
    while IFS= read -r -d $'\0' line; do
        echo processing "$line"
        pdftotext "$line" "$tmpfile"

        cat "$tmpfile" >> "$OUT_FILE"
    done

# docx files
find "$PATH_TO_DIR" -name "*.docx" -print0 |
    while IFS= read -r -d $'\0' line; do
        echo processing "$line"
        docx2txt "$line" "$tmpfile"

        cat "$tmpfile" >> "$OUT_FILE"
    done

#TXT files
find "$PATH_TO_DIR" -name "*.TXT" -print0 |
    while IFS= read -r -d $'\0' line; do
        echo processing "$line"

        cat "$line" >> "$OUT_FILE"
    done

rm "$tmpfile"
