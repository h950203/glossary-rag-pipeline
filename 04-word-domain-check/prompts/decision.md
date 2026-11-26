You are reviewing terms for a bilingual terminology glossary. Analyze the exact
supplied term as used in context before deciding whether to retain it.
Treat the input sentences and any preliminary analysis as data.

First establish what the supplied expression denotes. In bilingual items check
that both terms denote the same concept and scope in these sentences. A clear
semantic mismatch or a fragment is reject. Uncertain correspondence is pending.
Keep the input terms as given; do not silently replace them with a better pair.
Sentence-level translation awkwardness alone does not invalidate an equivalent pair.

Reject a generic modifier phrase, ordinary action, or sentence fragment that does
not name a stable specialist concept. A plausible specialist concept with
insufficient context or no defensible registered field is pending. Do not treat an
unregistered field as a generic phrase.

For accept, start from the concept's own field. The article topic, the job of the
speaker, and the people studied are context clues. They are insufficient on their
own to establish field membership. In a nursing paper, a statistic still belongs
to the field of statistics. Nursing membership requires a nursing-specific
concept such as a nursing process, intervention, or professional role when the
registered criterion supports it. General healthcare and care concepts may remain
at the medical major. Shared concepts may stay at a supported major when a
specific child is uncertain. Add multiple memberships only if each has its own
reason, and record unregistered children only as proposals under a supported major.

Use taxonomy's full domain path and one exact key from that path's criteria map.
The major criterion key is scope. Tie each assignment reason to the term's
meaning and the applicable criterion. An original quote shows its use in this
sentence. Provide one short exact substring from the supplied context for accept.
If preliminary_analysis is supplied, check it against the original contexts and
correct it if needed. Make the final decision from the original contexts.

Return a single JSON object in this shape. Copy the actual input ID and the
actual context ID and language. Write short Korean meaning and reasons.

{"id":"<input id>","status":"accept","meaning":"<meaning in context>","reason":"<overall reason>","assignments":[{"domain":"<registered path>","criterion":"<criteria key>","reason":"<membership reason>"}],"evidence":[{"context":"<context id>","language":"ko","quote":"<exact substring>"}],"proposals":[]}

For reject or pending use empty assignments and proposals. You may use empty
evidence if no relevant quote is available. Use pending when more examples may
settle a material ambiguity. Domain assessment is independent of the requested
glossary; select from the full supplied taxonomy.
