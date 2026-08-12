# Pressure-Field Coordination Paper - Build System
#
# All LaTeX build artifacts go to $(BUILDDIR); only final PDFs land in this
# directory. latexmk runs pdflatex/bibtex exactly as many times as needed.

MAIN     = rodriguez-pressure-field-coordination-2026
LETTERS  = cover-letter response-to-editor information-sheet
BUILDDIR = .build
LATEXMK  = latexmk -pdf -interaction=nonstopmode -outdir=$(BUILDDIR)

.PHONY: all paper letters clean distclean watch

all: paper

# Main manuscript
paper:
	$(LATEXMK) $(MAIN).tex
	@cp $(BUILDDIR)/$(MAIN).pdf .
	@rm -f *-eps-converted-to.pdf

# Ancillary submission documents
letters:
	$(LATEXMK) $(addsuffix .tex,$(LETTERS))
	@cp $(addprefix $(BUILDDIR)/,$(addsuffix .pdf,$(LETTERS))) .
	@rm -f *-eps-converted-to.pdf

# Remove the build directory and any stray artifacts left in this directory
clean:
	rm -rf $(BUILDDIR)
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
	rm -f *.synctex.gz *.bcf *.run.xml *-eps-converted-to.pdf

# Clean everything including the generated PDFs (figures are untouched)
distclean: clean
	rm -f $(MAIN).pdf $(addsuffix .pdf,$(LETTERS))

# Rebuild on change; the live PDF is $(BUILDDIR)/$(MAIN).pdf while watching
watch:
	latexmk -pdf -pvc -interaction=nonstopmode -outdir=$(BUILDDIR) $(MAIN).tex
