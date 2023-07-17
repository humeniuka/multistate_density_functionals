# Running
#  > make
# will build the pdf.

all: main.pdf

%.pdf: %.tex references.bib   build_always
	pdflatex $<
	bibtex $*
	pdflatex $<
	pdflatex $<

# This target forces the pdf document to be compiled always even if it's up-to-date
build_always:
	echo "Building PDF"

clean:
	rm -f  acs-main.bib main.aux main.bbl main.blg main.log main.pdf
