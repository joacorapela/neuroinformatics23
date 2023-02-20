all: rapela_ws3.pdf


%.pdf: %.tex
	pdflatex $<
	bibtex $*.aux
	pdflatex $<
	pdflatex $<
