thesis.pdf: bibliography.bib thesis.tex chapters/*.tex frontback/*.tex 
	pdflatex -shell-escape thesis
	bibtex thesis
	pdflatex -shell-escape thesis
	pdflatex -shell-escape thesis

clean:
	rm -f *.lot *.lof *.lol *.toc *.log *.out *.aux *.blg *.bbl thesis.pdf chapters/*.aux frontback/*.aux 

rebuild: clean thesis.pdf
