# Running
#
#  > make
# will build the pdf with the notes.
#
# > make main_long_version.pdf
# > make main_short_version.pdf
# will build the long or short versions of the manuscript.
#
# > make supporting_information_long_version.pdf
# > make supporting_information_short_version.pdf
# will build the supporting information for the long or short manuscript.
#

all: notes.pdf

%.pdf: %.tex references.bib   build_always
	pdflatex $<
	bibtex $*
	pdflatex $<
	pdflatex $<

# This target forces the pdf document to be compiled always even if it's up-to-date
build_always:
	echo "Building PDF"

# Remove Latex files that can be regenerated.
clean:
	rm -f  acs-notes.bib notes.aux notes.bbl notes.blg notes.log notes.pdf

	rm -f  acs-main_long_version.bib main_long_version.aux main_long_version.bbl main_long_version.blg main_long_version.log main_long_version.pdf
	rm -f  acs-main_short_version.bib main_short_version.aux main_short_version.bbl main_short_version.blg main_short_version.log main_short_version.pdf

	rm -f  acs-supporting_information_long_version.bib supporting_information_long_version.aux
	rm -f  supporting_information_long_version.bbl supporting_information_long_version.blg supporting_information_long_version.log supporting_information_long_version.pdf

	rm -f  acs-supporting_information_short_version.bib supporting_information_short_version.aux
	rm -f  supporting_information_short_version.bbl supporting_information_long_version.blg supporting_information_long_version.log supporting_information_long_version.pdf
