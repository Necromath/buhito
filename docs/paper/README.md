# Paper methods companion

methods.tex is a standalone subsection and two algorithms for the existing paper.
It describes default train-only, ranked-prefix MDL selection and frozen
transformation, including empty-dictionary outcomes and the exact lossless scope.

For Overleaf, upload methods.tex and add these preamble packages if the template
does not already provide them:

~~~latex
\usepackage{amsmath}
\usepackage{algorithm}
\usepackage{algpseudocode}
~~~

Insert the file at the intended methods section with:

~~~latex
\input{methods}
~~~

The surrounding paper's original-representation/feature-adapter experiment
definition still needs to be aligned with its actual experiments. These methods
do not assert that a compressed predictor has been implemented.

preview.tex provides a minimal compilation check. From this directory:

~~~bash
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
~~~

Keep generated preview outputs outside the tracked source tree. The private
Overleaf project's URL/source was not part of the repository; these files can
be inserted without depending on that project's availability.
