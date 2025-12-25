// NeurIPS-style template for Typst
// Based on NeurIPS 2024 formatting guidelines

#let neurips(
  title: none,
  authors: (),
  abstract: none,
  keywords: (),
  body,
) = {
  // Document setup
  set document(title: title)
  set page(
    paper: "us-letter",
    margin: (x: 1.5in, y: 1in),
    numbering: "1",
  )

  // Text setup - NeurIPS uses Times-like font
  set text(
    font: "New Computer Modern",
    size: 10pt,
  )

  // Paragraph setup
  set par(
    justify: true,
    leading: 0.55em,
    first-line-indent: 1.5em,
  )

  // Heading styles
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    set text(size: 12pt, weight: "bold")
    set par(first-line-indent: 0em)
    v(1em)
    it
    v(0.5em)
  }
  show heading.where(level: 2): it => {
    set text(size: 10pt, weight: "bold")
    set par(first-line-indent: 0em)
    v(0.8em)
    it
    v(0.4em)
  }
  show heading.where(level: 3): it => {
    set text(size: 10pt, weight: "bold", style: "italic")
    set par(first-line-indent: 0em)
    v(0.6em)
    it
    v(0.3em)
  }

  // Link styling
  show link: set text(fill: rgb("#0066cc"))

  // Math equation numbering
  set math.equation(numbering: "(1)")

  // Title block
  align(center)[
    #v(0.5in)
    #text(size: 17pt, weight: "bold")[#title]
    #v(1em)

    #if authors.len() > 0 {
      for author in authors {
        text(size: 10pt)[#author.name]
        if "affiliation" in author {
          text(size: 9pt)[ \ #author.affiliation]
        }
        if "email" in author {
          text(size: 9pt, style: "italic")[ \ #author.email]
        }
        v(0.5em)
      }
    }
    #v(1em)
  ]

  // Abstract
  if abstract != none {
    set par(first-line-indent: 0em)
    align(center)[
      #block(width: 85%)[
        #text(weight: "bold")[Abstract]
        #v(0.3em)
        #text(size: 9pt)[#abstract]
      ]
    ]
    v(1em)
  }

  // Keywords
  if keywords.len() > 0 {
    set par(first-line-indent: 0em)
    align(center)[
      #block(width: 85%)[
        #text(size: 9pt)[*Keywords:* #keywords.join(", ")]
      ]
    ]
    v(1em)
  }

  // Main body
  body
}

// Utility functions for common elements

#let theorem(name: none, body) = {
  block(
    width: 100%,
    inset: 8pt,
    stroke: (left: 2pt + gray),
  )[
    #if name != none {
      [*Theorem* (#name). ]
    } else {
      [*Theorem.* ]
    }
    #emph(body)
  ]
}

#let definition(name: none, body) = {
  block(
    width: 100%,
    inset: 8pt,
    stroke: (left: 2pt + gray),
  )[
    #if name != none {
      [*Definition* (#name). ]
    } else {
      [*Definition.* ]
    }
    #body
  ]
}

#let algorithm(name: none, body) = {
  figure(
    kind: "algorithm",
    supplement: [Algorithm],
    block(
      width: 100%,
      inset: 10pt,
      stroke: 1pt + gray,
      radius: 4pt,
    )[
      #if name != none {
        align(center)[*#name*]
        v(0.5em)
      }
      #body
    ],
  )
}
