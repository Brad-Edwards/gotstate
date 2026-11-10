# 24 Writing guide

Use ASD-STE100 Issue 9 style for the documents of this repository.
Write for a reader who needs to find a fact, make a decision, or complete a task.

<a id="document-structure"></a>

## 24.1 Document structure

Give each document one purpose.
Put its purpose or decision first.
Use the numbered [document index](00-contents.md) to find the owner of each subject.

- Number document titles and section headings.
- Prefix each filename with its document number, using at least two digits: `22-engineering-standards.md`.
- Number procedures, tables, and figures within each document.
- Leave ordinary paragraphs and reference lists unnumbered.
- Use at most four heading levels.
- Use the same term for the same concept.
- Link to the owning document instead of repeating its rules.

Chapters 1 to 12 follow the [arc42](https://arc42.org/overview) sections, and diagrams use the [C4 model](https://c4model.com/) levels.

<a id="sentences-and-paragraphs"></a>

## 24.2 Sentences and paragraphs

Name the actor and action.
Use active voice and familiar words.
Keep one instruction in each sentence.

Use at most 20 words in a procedure step and 25 words in a descriptive sentence.
Keep each paragraph to one subject and at most three sentences.
Use lists for parallel facts and tables for comparisons.

Use `must` for a requirement and `can` for an ability or a possibility.
State the condition for optional behavior.
Use the full form of contractions.

Put each sentence on a separate source line where practical.
Keep code, commands, identifiers, and exact diagnostic messages in code spans or blocks.
Keep link labels short and specific.

<a id="terms-and-abbreviations"></a>

## 24.3 Terms and abbreviations

Define each project term where a document first uses it.
Use the approved term in headings, tables, diagrams, and prose.
Expand an abbreviation at its first use in a document.

Technical names identify products, protocols, fields, and components.
Technical verbs describe specific operations, such as compile, authenticate, encrypt, and validate.
Keep their meanings consistent across documents.

<a id="decisions-status-and-evidence"></a>

## 24.4 Decisions, status, and evidence

State document status once when it affects the reader's decision.
Put unresolved choices in one decision or gap table with an owner and next action.
Keep assessment results in their records.

Describe the required behavior directly.
Keep a warning only when it changes an action or prevents a specific error.
Remove repeated explanations of what a design, diagram, test, or tool does not prove.

Preserve conditions, numerical limits, security boundaries, and failure behavior during editing.
A shorter sentence must retain the same requirement.
Keep historical source revisions and citations intact.

<a id="automated-checks"></a>

## 24.5 Automated checks

The repository stores its Vale rules in `styles/STE`.
The rules use original project expressions and plain-English substitutions.
They do not redistribute the ASD dictionary or import an external style package.

**Table 24.1: Automated checks**

| Check | Enforcement |
| --- | --- |
| Sentence and procedure length | Vale limits descriptions to 25 words and list items to 20 words. |
| Paragraph length | Vale permits at most three sentences per paragraph. |
| Word choice | Vale checks plain-English substitutions. |
| Contractions and repetition | Vale reports contractions and repeated adjacent words. |
| Ambiguity | Vale reports open-ended lists, combined alternatives, and vague qualifiers. |
| Voice | Vale detects explicit passive constructions that name the actor after the action. |
| References | MkDocs strict mode checks local files, section anchors and the navigation. |
| Numbering, captions and meaning | Review checks filename prefixes, section order, captions, technical accuracy, approved word meanings and one instruction per sentence. |

Vale uses a conservative token count.
Review applies the standard's counting rules for technical names, measurements, and quoted text.
The reviewer checks the ASD-STE100 rules that no automated check covers.

<a id="check-and-read-the-docs"></a>

## 24.6 Check and read the documents

**Procedure 24.1: Check and read the documents**

1. Run `make vale-lint FILES=docs/24-writing-guide.md` to check a selected document.
2. Check the document number, captions, and links while editing.
3. Run `make docs` to build the local reader.
4. Run `make docs-serve` to read the documents at the displayed local address.
5. Require passing CI results for all documents before merge.

Check a proposed issue body with `make vale-lint FILES=body.md`.
This command uses the repository's trusted Vale configuration.

`mkdocs.yml` owns document numbers and reader navigation.
Keep document numbers and named anchors stable when headings change.
Update links and navigation when a filename changes.

<a id="sources"></a>

## 24.7 Sources

- [ASD-STE100 official standard and access](https://www.asd-ste100.org/).
- [ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).
- [ASD-STE100 questions and vocabulary guidance](https://www.asd-ste100.org/STE_faq.html).
- [Vale checks and scopes](https://docs.vale.sh/topics/scopes).
- [MkDocs document layout](https://www.mkdocs.org/user-guide/writing-your-docs/).
