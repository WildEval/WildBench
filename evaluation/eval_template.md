# Instruction 

You are an expert evaluator. Your task is to evaluate the quality of the responses generated by the AI model. 
We will provide you with the user query and a pair of AI-generated responses (Response A and Response B).
Also, we will provide you with a checklist and rules to guide your evaluation.

# Conversation between User and AI

## History
```
{$history}
``` 

## User Query
```
{$user_query}
```

## Response A
```
{$candidate_A}
```

## Response B
```
{$candidate_B}
```

# Evaluation   

## Checklist 

```
{$checklist}
```

## Rules 

You should compare the above two responses based on the user queries and the above checklist.
You should use a few short sentences to briefly show your assessment according to the checklist.
Finally, you have three choices to give final assessment: ["A", "B", "tie"].
- Select `A` only when Response A is better than Response B.
- Select `B` only when Response B is better than Response A.
- Select `tie` when Response A and B are of the same quality. Please use this choice sparingly.
- Please do not overly prefer the longer response. The length of the response should not be the only factor to consider.

## Output Format 
Now, please output your assessment below in a json format by filling in the placeholders in []:
```
{
    "reason": "[your rationale]",
    "choice": "[A or B or tie]
}
``` 