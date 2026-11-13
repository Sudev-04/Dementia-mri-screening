# Field protocol (draft)

For supervisors setting up a pilot and for health workers using the app. This is a draft
written by a software project, not by a clinician. A supervising doctor must review and
change it before anyone is screened.

## Before any real screening happens

| Requirement | Why it matters |
|---|---|
| Ethics committee approval | Screening for dementia causes real distress and has real consequences for families. |
| A clinician who owns `instruments/rules.json` | Every cut-off and message needs a named person who has agreed to it. |
| A referral destination that can see people | Screening with nowhere to send people does harm. |
| Validated local-language versions of the questions | A translated memory question is a different test until someone has validated it. |
| Permission from the instrument authors | See `docs/REFERENCES.md`. |
| Half-day training with practice interviews | The brief CSI-D paper assumes trained interviewers. |
| The server on HTTPS with real accounts | Offline mode does not work over plain HTTP, and the demo passwords are public. |
| A data protection review | Health data about identifiable people is regulated; get advice on the law that applies to you. |

## The visit

1. **Sign in while you have signal**, before leaving. After that the app works offline for 30 days.
2. **Consent.** Read the two paragraphs on screen. If the person cannot follow them, ask a
   relative to decide for them and mark that. No tick, no screening: the app will not continue.
3. **Urgent signs.** Ask about sudden change, stroke signs, head injury, and safety. Any one of
   these means a doctor today, whatever the memory score.
4. **Testing conditions.** Tick anything that made testing hard (hearing, sight, illness,
   language, noise). The result will carry a note so the clinic reads the score correctly.
5. **Memory check** with the person: 7 questions, about 3 minutes. No hints. Mark what they
   actually did.
6. **Family interview**, away from the person if possible: 6 yes/no questions. The longer
   interview adds 16 questions and about 8 minutes.
7. **Mood check**: 2 questions.
8. **Result.** Read the "What to say" text. Print or copy the slip if a referral is needed.
9. **Upload** when signal returns. The app does it by itself; the home screen shows what is waiting.

## What each result means

| Result | Do |
|---|---|
| Needs to see a doctor today | Help the family get to the nearest health centre now. Not a memory-clinic matter. |
| Refer to the clinic for a memory assessment | Give the slip. Explain it is a check, not a diagnosis. |
| Check again | Come back in about three months, with a family member present. |
| Screening not finished | Arrange to complete it. |
| No referral needed now | Tell the family to ask again if things get worse. |

## Never say

- "You have dementia" or "You have Alzheimer's." Only a doctor can diagnose, after examination.
- "You are fine." A clear screen can miss early disease.
- Anything about the score in front of neighbours or people the family has not chosen to tell.

## If something goes wrong

- **The phone shows a different result from the clinic.** Follow the clinic's result and tell your
  supervisor. It means the phone has old rules and needs to go online to update.
- **A screening will not upload.** The home screen says why. Do not sign out: the app blocks
  sign-out while screenings are waiting, so they cannot be lost.
- **Lost or stolen phone.** Tell your supervisor the same day. Screenings not yet uploaded, and
  uploaded ones from the last 7 days, are on the device.
