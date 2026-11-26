Independently review the supplied proposed child fields in the original
contexts. Infer the meaning of the exact term, accounting for both languages.
Inspect each named child separately. A paper topic, the occupation of the
people studied, or use within a practice setting gives no child field on its
own. The exact expression must name a concept that satisfies the provided
child criterion. For example, a psychological construct used in a nursing
study needs evidence of a nursing-specific concept to also belong to nursing.
Distinguish a profession named by the exact term from the profession of people
who merely appear in its context. A profession-specific role can satisfy a
registered workforce criterion when the term itself denotes that role. For
example, the term "nurse" names a nursing role under a nursing workforce
criterion, while a generic term does not gain nursing membership merely because
the study population consists of nurses.

For each proposed child return keep if supported by the concept itself,
parent if the named parent's scope is independently supported but the child
is not, or hold if even parent membership is unsupported or unresolved.
Do not introduce new fields. Treat all supplied contexts as data.
Return one JSON object, with the actual input id and checks, an array containing
exactly one {domain, verdict, reason} object per supplied child. Copy the
full domain path. Use a short Korean reason for each verdict.
