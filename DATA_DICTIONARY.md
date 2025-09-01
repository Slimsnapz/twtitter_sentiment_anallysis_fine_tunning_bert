# Data Dictionary

Source file: `twitter_multi_class_sentiment.csv`

Total rows: **16000**

## Columns

- **text** (`object`) — missing: 0
- **label** (`int64`) — missing: 0
- **label_name** (`object`) — missing: 0

## Sample rows (first 5)
```json
[
  {
    "text": "i didnt feel humiliated",
    "label": 0,
    "label_name": "sadness"
  },
  {
    "text": "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake",
    "label": 0,
    "label_name": "sadness"
  },
  {
    "text": "im grabbing a minute to post i feel greedy wrong",
    "label": 3,
    "label_name": "anger"
  },
  {
    "text": "i am ever feeling nostalgic about the fireplace i will know that it is still on the property",
    "label": 2,
    "label_name": "love"
  },
  {
    "text": "i am feeling grouchy",
    "label": 3,
    "label_name": "anger"
  }
]
```

## Label column: **label**
Class distribution:
- `1`: 5362 examples
- `0`: 4666 examples
- `3`: 2159 examples
- `4`: 1937 examples
- `2`: 1304 examples
- `5`: 572 examples