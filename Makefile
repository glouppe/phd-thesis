thesis.pdf: bibliography.bib thesis.tex 
	pdflatex -shell-escape thesis
	bibtex thesis
	pdflatex -shell-escape thesis
	pdflatex -shell-escape thesis

clean:
	rm -f *.lot *.lof *.lol *.toc *.log *.out *.aux *.bbl *.blg thesis.pdf chapters/*.aux frontback/*.aux 
