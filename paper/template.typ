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
  // New Computer Modern has generous x-height; settings tuned accordingly
  set text(
    font: "New Computer Modern",
    size: 10pt,
    // Kerning: enable pairwise spacing adjustments (on by default, explicit for clarity)
    kerning: true,
    // Ligatures: enable standard ligatures (fi, fl, ff, ffi, ffl) for serif text
    ligatures: true,
    // Numbers: lining figures for academic consistency; tabular handled per-context
    number-type: "lining",
    // Tracking: 0pt for body text preserves designer's intended letter-fit
    tracking: 0pt,
    // Word spacing: slight increase (2%) improves justified text color
    spacing: 102%,
    // Top/bottom edge: explicit for predictable baseline calculations
    top-edge: "cap-height",
    bottom-edge: "baseline",
    // Overhang: allow punctuation to hang slightly into margins for optical alignment
    overhang: true,
  )

  // Paragraph setup
  // Leading: 0.58em at 10pt ≈ 15.8pt baseline-to-baseline (~158% line-height)
  // This balances density for academic text with readability
  set par(
    justify: true,
    leading: 0.58em,
    first-line-indent: 1.5em,
    // Spacing between paragraphs: 0.8em provides clear separation without excess whitespace
    spacing: 0.8em,
  )

  // Heading styles
  // Bold text benefits from slight positive tracking to counter optical cramping
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    set text(
      size: 12pt,
      weight: "bold",
      // +0.02em tracking opens up bold letterforms at display size
      tracking: 0.02em,
    )
    set par(first-line-indent: 0em)
    v(1em)
    it
    v(0.5em)
  }
  show heading.where(level: 2): it => {
    set text(
      size: 10pt,
      weight: "bold",
      // Slightly less tracking at text size
      tracking: 0.015em,
    )
    set par(first-line-indent: 0em)
    v(0.8em)
    it
    v(0.4em)
  }
  show heading.where(level: 3): it => {
    set text(
      size: 10pt,
      weight: "bold",
      style: "italic",
      // Italic+bold: modest tracking to maintain legibility
      tracking: 0.01em,
    )
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
    // Title: no justification or hyphenation—prevents rivers in display text
    #set par(justify: false)
    #text(size: 17pt, weight: "bold", tracking: 0.03em, hyphenate: false)[#title]
    #v(1em)

    #if authors.len() > 0 {
      for author in authors {
        text(size: 10pt)[#author.name]
        if "affiliation" in author {
          // Affiliations: slightly tighter leading for compactness
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
        // "Abstract" label: small caps alternative via tracking
        #text(weight: "bold", tracking: 0.05em)[Abstract]
        #v(0.3em)
        // Abstract body: 9pt needs proportionally more leading for readability
        #set par(leading: 0.62em)
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
        // Keywords at 9pt with consistent leading
        #set par(leading: 0.62em)
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
