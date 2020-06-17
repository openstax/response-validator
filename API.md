
|Route|Response|Purpose|
|-----|--------|-------|
|`/ping`| `pong`| Determining that the validation service is operational.|
|`/version` or `/rev.txt`| version string (i.e. 2.3.0)|What version of service is installed|
|`/status`| json response (see below)| Detailed service info (extended version, start time) and datasets|


Here is the `/status` response for a server started on Oct 15, with a clean install of version 2.4.0,
vocabularies for 5 books, and existing feature weight set IDs loaded:

```json
{
  "datasets": {
    "books": [
      {
        "name": "Biology 2e",
        "vuid": "8d50a0af-948b-4204-a71d-4826cba765b8@15.45"
      },
      {
        "name": "College Physics for AP® Courses",
        "vuid": "8d04a686-d5e8-4798-a27d-c608e4d0e187@26.1"
      },
      {
        "name": "College Physics with Courseware",
        "vuid": "405335a3-7cff-4df2-a9ad-29062a4af261@7.53"
      },
      {
        "name": "Introduction to Sociology 2e",
        "vuid": "02040312-72c8-441e-a685-20e9333f3e1d@10.1"
      },
      {
        "name": "Biology for AP® Courses",
        "vuid": "6c322e32-9fb0-4c4d-a1d7-20c95c5c7af2@18.4"
      }
    ]
  },
  "started": "Tue Oct 15 16:09:23 2019",
  "version": {
    "date": "2019-10-15T14:40:38-0500",
    "dirty": false,
    "error": null,
    "full-revisionid": "463fc5ef4c9d8c37aa600720c8bc814dfa44557c",
    "version": "2.4.0"
  }
}
```
The `datasets` at the top list the books that have their vocabularies loaded and available.

### Dataset APIs

The following routes all serve JSON formatted representations of the datasets
used by the validator to make its validity determinations. Books,
exercise vocabularies and the weights used to combine the feature values (feature coefficients) are available.

|Route|Response
|-----|--------
/datasets | list of classes of datasets available
/datasets/books| list of books
/datasets/books/`<book-vuid>`| Data for a single book
/datasets/books/`<book-vuid>`/pages | list of pages for a single book
/datasets/books/`<book-vuid>`/pages/`<page-vuid>` | Data for a single page - ID, innovation words, list of questions
/datasets/books/`<book-vuid>`/vocabularies | list of vocabularies for a single book
/datasets/books/`<book-vuid>`/vocabularies/domain | list of non-common words in the book
/datasets/books/`<book-vuid>`/vocabularies/innovation | lists of novel words in each page of the book, by page
/datasets/books/`<book-vuid>`/vocabularies/innovation/`<page-vuid>` | list of novel words for a specific page in the book
/datasets/questions| list of questions
/datasets/questions/`<question-vid>`| Data for a single question (id + vocabularies)
/datasets/feature_weights | list of existing feature weight set IDs
/datasets/feature_weights/`<fw-id>` | Data for a single set of feature weights

### Processing APIs

Route|Propose|Use
---|---|---
/import|load a book and associated exercises| POST a tutor ecosystem YAML file
/train|find best-fit feature coefficents| POST a response training set


#### TODO:

- store feature coefficent sets, return IDs
- additional data APIs for downloading ~exercise vocabularies and~ feature cofficient sets
- Currently there is no security for this app (anything can call it).  I am not sure how this is usually handled in Tutor but it should not be too difficult to add an api key or similar security measures.
- Depending on UX, we may want to return more granular information about the response rather than a simple valid/non-valid label.  We can modify this easily enough as the need arises.

