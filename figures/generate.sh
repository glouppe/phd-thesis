latex $1.tex
dvips $1.dvi
ps2pdf $1.ps
pdfcrop $1.pdf
rm $1.aux $1.dvi $1.log $1.ps $1.pdf
mv $1-crop.pdf $1.pdf
