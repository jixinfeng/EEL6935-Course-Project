#!/bin/bash

ERROR="Too few arguments : no file name specified"
[[ $# -eq 0 ]] && echo $ERROR && exit # no args? ... print error and exit

if [ -f $1.tex ];then
    pdflatex $1
    bibtex $1
    pdflatex $1
    pdflatex $1
    rm *.lof *.lot *.out *.toc *.log *.aux *.bbl *.blg *.bak *.sav *.dvi *.gz
fi
