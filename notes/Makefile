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
	rm -f  acs-main_long_version.bib main_long_version.aux main_long_version.bbl main_long_version.blg main_long_version.log main_long_version.pdf
	rm -f  acs-supporting_information_long_version.bib supporting_information_long_version.aux
	rm -f  supporting_information_long_version.bbl supporting_information_long_version.blg supporting_information_long_version.log supporting_information_long_version.pdf
