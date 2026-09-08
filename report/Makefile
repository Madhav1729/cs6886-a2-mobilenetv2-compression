# Build the report:  make        (runs pdflatex twice for refs)
#                    make clean  (remove build artifacts, keep the PDF)
report.pdf: report.tex figures/baseline_curves.png figures/parallel_coordinates.png
	pdflatex -interaction=nonstopmode report.tex
	pdflatex -interaction=nonstopmode report.tex
clean:
	rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz
.PHONY: clean
