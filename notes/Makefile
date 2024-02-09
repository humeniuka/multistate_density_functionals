# Running
#  > make
# will build the pdf.

all: notes.pdf

%.pdf: %.tex references.bib   build_always
	pdflatex $<
	bibtex $*
	pdflatex $<
	pdflatex $<

# This target forces the pdf document to be compiled always even if it's up-to-date
build_always:
	echo "Building PDF"

clean:
	rm -f  acs-notes.bib notes.aux notes.bbl notes.blg notes.log notes.pdf
	rm -f  acs-main.bib main.aux main.bbl main.blg main.log main.pdf
	rm -f  acs-supporting_information.bib supporting_information.aux
	rm -f  supporting_information.bbl supporting_information.blg supporting_information.log supporting_information.pdf
