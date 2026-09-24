# jev-gliner bench2 large suite: 7 families x 15 items = 105 items.
# ALL items hand-written for this benchmark (2026-09-18); items author = ground-truth author.
# No model-generated ground truth anywhere in this file.
# Split: *_01..*_03 = "tune" (GLiNER threshold selection ONLY); *_04..*_15 = "test".
# Intent: items + data published alongside the benchmark thread.

from typing import Any, NotRequired, TypedDict

FAMILIES = ["route", "conditions", "aspects", "entities", "severity", "verify", "records"]

ACTION_LABELS = ["read", "write", "delete", "summarize"]
CONFIRM_RULE = ("destructive_requires_confirmation: would carrying out this request destroy or "
                "irreversibly change existing state - deleting, overwriting, sending, deploying, or paying?")
COND_LABELS = ["billing_dispute", "contains_amount", "mentions_deadline", "escalated", "angry_tone"]
COND_PLAIN = {
    "billing_dispute": "a billing dispute (the sender contests a charge, invoice, fee, or bill)",
    "contains_amount": "a currency amount (a dollar figure like $58 or $1,200)",
    "mentions_deadline": "a deadline, due date, or timeframe for action",
    "escalated": "that the issue was ALREADY escalated (a prior ticket, supervisor, formal complaint, or another team)",
    "angry_tone": "an angry, hostile, or furious tone",
}
ASPECT_LABELS = ["camera", "performance", "battery", "display", "price"]
ENT_LABELS = ["company", "person", "product", "location"]
SEV_LEVELS = ["trivial", "minor", "moderate", "major", "critical"]


class Task(TypedDict):
    """One suite case: id, family, tune/test split, model input, ground truth."""

    id: str
    family: str
    split: str
    input: Any
    gt: Any
    distractors: NotRequired[Any]
    gt_rationale: str

TASKS: list[Task] = []

# ------------------------------------------------------------------ ROUTE (15)
# gt: action (read/write/delete/summarize) + confirm (destructive_requires_confirmation)
_route: list[tuple[str, dict[str, str], str]] = [
    ("Can you delete the temporary report file at /tmp/report_q3.pdf? It's taking up space.",
     {"action": "delete", "confirm": "yes"},
     "deletes an existing file: irreversible, confirm first"),
    ("Summarize the thread with the vendor about the SLA amendment - I need the key points before the 2pm call.",
     {"action": "summarize", "confirm": "no"},
     "read-only summary request; changes nothing"),
    ("Draft a reply to the invoice query but don't send it yet - I want to review it first.",
     {"action": "write", "confirm": "no"},
     "creates a draft, explicitly unsent: fully reversible"),
    ("Pull the last hour of error logs for the payments service and show me anything with status 500.",
     {"action": "read", "confirm": "no"},
     "pure read of logs"),
    ("Tear down the retired payments-staging namespace in the cluster - it's been idle since July.",
     {"action": "delete", "confirm": "yes"},
     "deletes a retired cluster namespace: irreversible removal of existing state"),
    ("Send the all-hands announcement about the office move to everyone now - it's 9am, best time to hit inboxes.",
     {"action": "write", "confirm": "yes"},
     "sends email to the whole company: cannot be unsent"),
    ("Run the payroll batch for this cycle - approve and submit it; accounting already signed off.",
     {"action": "write", "confirm": "yes"},
     "submits payroll: moves real money, irreversible"),
    ("Promote build 2026.9.17 to production - QA green-lit it an hour ago.",
     {"action": "write", "confirm": "yes"},
     "production deploy: changes live service state"),
    ("Create a new file called notes_2026-09-18.md in my Documents folder with today's agenda from the standup notes.",
     {"action": "write", "confirm": "no"},
     "creates a brand-new file; nothing existing is modified"),
    ("Search my inbox for invoices from Figma this quarter and tell me the total.",
     {"action": "read", "confirm": "no"},
     "search-and-report: read-only"),
    ("Deploy the analytics service to the staging cluster so QA can poke at it tomorrow.",
     {"action": "write", "confirm": "no"},
     "staging deploy: sandboxed, easily reversible"),
    ("Remove the old backups directory /var/backups/app-2024 - we keep seven years, anything older can go.",
     {"action": "delete", "confirm": "yes"},
     "deletes backup data: irreversible loss of existing state"),
    ("Open the September statement and check whether the AWS charge matches last month's.",
     {"action": "read", "confirm": "no"},
     "read-and-compare only"),
    ("Give me a two-line summary of the customer escalation thread from this morning.",
     {"action": "summarize", "confirm": "no"},
     "summary only"),
    ("Roll production back to yesterday's release - the new build is spiking 500s.",
     {"action": "write", "confirm": "yes"},
     "changes live production state under incident pressure; irreversible in effect"),
]
gt: Any  # loop target reused across families with different gt shapes
for i, (text, gt, rat) in enumerate(_route, 1):
    TASKS.append({"id": f"route_{i:02d}", "family": "route",
                  "split": "tune" if i <= 3 else "test",
                  "input": text, "gt": gt, "gt_rationale": rat})

# ------------------------------------------------------------- CONDITIONS (15)
# gt: yes/no per 5 fixed labels. >=4 items all-5-true; >=4 items <=2 true (balance asserted).
_cond: list[tuple[str, dict[str, str], str]] = [
    ("This is the THIRD time I'm writing about invoice #8821. You charged my card $349.00 twice for the same "
     "subscription. I opened ticket #5521 three weeks ago and your team closed it without doing anything. "
     "I need this refunded by Friday or I'm calling my bank and disputing the charge. Unbelievable.",
     {"billing_dispute": "yes", "contains_amount": "yes", "mentions_deadline": "yes", "escalated": "yes", "angry_tone": "yes"},
     "contests double charge; $349.00; 'by Friday'; prior ticket closed unresolved; furious"),
    ("Hi team - quick question on invoice INV-2091: the line item shows 12 seats but we only have 10 users. "
     "Can someone check the math? Not urgent, just want it right before renewal.",
     {"billing_dispute": "yes", "contains_amount": "no", "mentions_deadline": "no", "escalated": "no", "angry_tone": "no"},
     "contests a billing line item; 12/10 are seat counts not currency; 'before renewal' is not a deadline"),
    ("Hi! Loving the new dashboard. Quick thing: my September invoice shows $149 but my plan is $99/mo - "
     "I think I was billed the old price. Could you take a look? No rush.",
     {"billing_dispute": "yes", "contains_amount": "yes", "mentions_deadline": "no", "escalated": "no", "angry_tone": "no"},
     "contests billed price; $149/$99 figures; 'no rush' is explicitly no deadline"),
    ("Our October bill charged the unused add-on again - we cancelled it in August. Can you refund the $40 "
     "and get this fixed before our renewal on the 1st? Thanks for the help.",
     {"billing_dispute": "yes", "contains_amount": "yes", "mentions_deadline": "yes", "escalated": "no", "angry_tone": "no"},
     "contests charge for cancelled add-on; $40; 'before renewal on the 1st'"),
    ("FYI the annual renewal invoice lands next week - $4,800 total, due by October 15. Finance needs the PO "
     "attached before processing, so please send it over when you can.",
     {"billing_dispute": "no", "contains_amount": "yes", "mentions_deadline": "yes", "escalated": "no", "angry_tone": "no"},
     "routine invoice notice, nothing contested; $4,800; 'due by October 15'"),
    ("Reminder that the Q4 budget doc is due to leadership by this Thursday. Nothing needed from your side - "
     "just don't want it slipping.",
     {"billing_dispute": "no", "contains_amount": "no", "mentions_deadline": "yes", "escalated": "no", "angry_tone": "no"},
     "pure deadline reminder; no charge contested, no currency"),
    ("Passing this along to your team lead since we've gone back and forth three times on it already - the data "
     "export issue is now formally with your support manager per our contract's escalation path. Not upset, "
     "just want it tracked properly.",
     {"billing_dispute": "no", "contains_amount": "no", "mentions_deadline": "no", "escalated": "yes", "angry_tone": "no"},
     "formally escalated per contract path; explicitly 'not upset'; no billing, amount, or deadline"),
    ("I am absolutely done being polite about this. You've billed my company card $1,180 for a service we "
     "CANCELLED in June. I've already gone over your head to your VP of support and filed a complaint with "
     "the BBB. Fix the billing and refund me by end of week or we're done as a customer.",
     {"billing_dispute": "yes", "contains_amount": "yes", "mentions_deadline": "yes", "escalated": "yes", "angry_tone": "yes"},
     "contests charge for cancelled service; $1,180; 'end of week'; VP + BBB complaint; 'done being polite'"),
    ("Thanks for the quick turnaround on the integration question yesterday - the webhook replay tip fixed it "
     "on the first try. Have a great week!",
     {"billing_dispute": "no", "contains_amount": "no", "mentions_deadline": "no", "escalated": "no", "angry_tone": "no"},
     "pure thank-you note: none of the five conditions present"),
    ("Saw the new pricing page - the Pro tier going to $18/mo makes this an easy yes for our whole team. Just "
     "confirming there's no change to our current contract's rate?",
     {"billing_dispute": "no", "contains_amount": "yes", "mentions_deadline": "no", "escalated": "no", "angry_tone": "no"},
     "$18/mo mentioned; a pre-emptive question, not a contest of any existing charge"),
    ("Second time I'm being bounced between your billing team and 'technical support' with nobody owning this. "
     "I was told THREE times someone would call me back and nobody did. Do your jobs.",
     {"billing_dispute": "no", "contains_amount": "no", "mentions_deadline": "no", "escalated": "yes", "angry_tone": "yes"},
     "bounced between teams + broken callback promises = escalated; hostile tone; no contested charge"),
    ("This is now the fourth month in a row we've been double-billed for the enterprise seats - $2,140 extra "
     "every cycle, and I'm sick of it. Ticket #9917 from August was escalated to a supervisor and then just "
     "went quiet. Get the refund processed and the billing fixed by Wednesday, otherwise legal will take this "
     "over. This is beyond ridiculous.",
     {"billing_dispute": "yes", "contains_amount": "yes", "mentions_deadline": "yes", "escalated": "yes", "angry_tone": "yes"},
     "contests recurring double-billing; $2,140; 'by Wednesday'; supervisor escalation went quiet; 'sick of it'"),
    ("The late fee on invoice INV-7734 is wrong - I paid on the due date (the bank transfer cleared that "
     "morning). Please waive it and correct the account before the next statement generates.",
     {"billing_dispute": "yes", "contains_amount": "no", "mentions_deadline": "yes", "escalated": "no", "angry_tone": "no"},
     "contests a late fee; no dollar figure anywhere; 'before the next statement generates'"),
    ("Escalating on behalf of customer Cinematrix (case #55102, formally escalated through their CSM): they've "
     "been charged the $399 enterprise rate since May but their contract says $299. They expect the "
     "back-difference refunded and the rate corrected before their October 1 auto-renewal. Treat as priority - "
     "no tone issues on their side, just very firm.",
     {"billing_dispute": "yes", "contains_amount": "yes", "mentions_deadline": "yes", "escalated": "yes", "angry_tone": "no"},
     "contested rate; $399/$299; 'before October 1'; formal escalation via CSM; explicitly firm-not-angry"),
    ("Are you KIDDING me with this? Third overcharge in two months - $96.50 this time for 'priority support' I "
     "never signed up for. I've asked twice already and got form replies, so this is going to your director "
     "and I'm filing a chargeback today unless the $96.50 is back in my account by tomorrow 5pm. Absolutely "
     "furious.",
     {"billing_dispute": "yes", "contains_amount": "yes", "mentions_deadline": "yes", "escalated": "yes", "angry_tone": "yes"},
     "contests unauthorized add-on charge; $96.50; 'tomorrow 5pm'; director + chargeback; 'Absolutely furious'"),
]
for i, (text, gt, rat) in enumerate(_cond, 1):
    TASKS.append({"id": f"cond_{i:02d}", "family": "conditions",
                  "split": "tune" if i <= 3 else "test",
                  "input": text, "gt": gt, "gt_rationale": rat})

# --------------------------------------------------------------- ASPECTS (15)
# gt: which of 5 fixed labels the review expresses an opinion on. Present-count varies 1..5.
_asp: list[tuple[str, list[str], str]] = [
    ("Great camera quality, decent performance, but poor battery life and the display scratches far too easily.",
     ["camera", "performance", "battery", "display"],
     "opinions on camera, performance, battery, display; price absent"),
    ("The price is honestly unbeatable for what you get.",
     ["price"],
     "single opinion: price"),
    ("Battery lasts two full days of heavy use, and the camera finally rivals flagship phones.",
     ["battery", "camera"],
     "battery + camera praised; others absent"),
    ("Blazing fast performance even with a dozen apps open, gorgeous display, and at $699 it undercuts everything "
     "similar. Camera is merely okay, and battery gets through a workday with a top-up.",
     ["camera", "performance", "battery", "display", "price"],
     "all five: performance, display, price, camera ('merely okay' is still an opinion), battery"),
    ("Screen is stunning outdoors and indoors - no complaints there.",
     ["display"],
     "display only"),
    ("Choppy performance when scrolling and the phone heats up, but I can't argue with the price.",
     ["performance", "price"],
     "performance criticized, price praised"),
    ("Camera is the star: sharp photos even at night. Battery is the letdown - barely four hours of screen time.",
     ["camera", "battery"],
     "camera praised, battery criticized"),
    ("For $1,099 I expected more than this mediocre screen and an aging chip.",
     ["price", "display", "performance"],
     "value-for-money complaint (price), 'mediocre screen' (display), 'aging chip' (performance); battery absent"),
    ("Absolutely terrible battery - dies by 2pm with light use.",
     ["battery"],
     "battery only"),
    ("The display calibrates beautifully and the camera app is intuitive, but performance lags on video edits and "
     "the price is a tough sell next to the competition.",
     ["camera", "performance", "display", "price"],
     "camera, display, performance, price evaluated; battery absent"),
    ("Solid mid-ranger: nothing about the camera, screen, or speed will wow you, but nothing offends either.",
     ["camera", "display", "performance"],
     "'nothing will wow you' still evaluates camera, display, performance; battery/price absent"),
    ("You pay a premium here - it costs $200 more than the base model - but the best-in-class battery life "
     "justifies it. The display got no refresh and the camera hardware is identical to last year's.",
     ["price", "battery", "display", "camera"],
     "price (premium), battery (praised), display + camera explicitly judged against last year; performance absent"),
    ("Performance is flawless - zero lag in two weeks of daily driving apps I use for work.",
     ["performance"],
     "performance only"),
    ("The battery and the display are the reasons to buy this.",
     ["battery", "display"],
     "battery + display"),
    ("Overpriced for the specs, the camera misses focus constantly, and the battery barely survives a morning. "
     "The only silver lining is the punchy display.",
     ["price", "camera", "battery", "display"],
     "price, camera, battery criticized; display praised; performance absent"),
]
for i, (text, gt, rat) in enumerate(_asp, 1):
    TASKS.append({"id": f"asp_{i:02d}", "family": "aspects",
                  "split": "tune" if i <= 3 else "test",
                  "input": text, "gt": gt, "gt_rationale": rat})

# -------------------------------------------------------------- ENTITIES (15)
# gt spans per label + 2 distractor candidate strings per text (real substrings fitting a DIFFERENT
# label). Jev sees gt spans + distractors + 'none' as choice candidates; scored against gt only.
_ents: list[tuple[Any, ...]] = [
    ("Amazon CEO Andy Jassy unveiled the Bedrock platform at the summit in Las Vegas.",
     {"company": ["Amazon"], "person": ["Andy Jassy"], "product": ["Bedrock"], "location": ["Las Vegas"]},
     {"company": ["Bedrock"], "person": ["CEO"], "product": ["Amazon"], "location": ["the summit"]},
     "straightforward one-per-label sentence"),
    ("Sundar Pichai opened the I/O keynote in Mountain View by showing off Gemini Nano running on Pixel devices "
     "built by Google.",
     {"company": ["Google"], "person": ["Sundar Pichai"], "product": ["Gemini Nano"], "location": ["Mountain View"]},
     {"company": ["Pixel"], "person": ["Pichai"], "product": ["Pixel devices"], "location": ["I/O keynote"]},
     "nested/adjacent spans; 'Pixel' is a product-like distractor for company"),
    ("Marta Vidal, who runs logistics at Maersk, said the new TerminalLink portal launched last month from their "
     "Copenhagen office.",
     {"company": ["Maersk"], "person": ["Marta Vidal"], "product": ["TerminalLink"], "location": ["Copenhagen"]},
     {"company": ["TerminalLink"], "person": ["who"], "product": ["logistics"], "location": ["office"]},
     "no famous entities; camel-cased product name"),
    ("We migrated our billing stack from Stripe to Adyen in March, and engineer Lena Hoff wrote up the whole saga "
     "on the company blog from our Stockholm hub.",
     {"company": ["Stripe", "Adyen"], "person": ["Lena Hoff"], "product": [], "location": ["Stockholm"]},
     {"company": ["billing stack"], "person": ["engineer"], "product": ["blog"], "location": ["hub"]},
     "TWO companies, zero products - counts vary by item"),
    ("During the Paris fireside, OpenAI's Sam Altman demoed SearchGPT and quipped that Dublin regulators keep him "
     "up at night.",
     {"company": ["OpenAI"], "person": ["Sam Altman"], "product": ["SearchGPT"], "location": ["Paris", "Dublin"]},
     {"company": ["SearchGPT"], "person": ["Altman"], "product": ["fireside"], "location": ["regulators"]},
     "TWO locations; partial-name distractor 'Altman' for person"),
    ("Hiroshi Tanaka from Sony's Tokyo team pitched the Afeela concept car to dealers in Osaka.",
     {"company": ["Sony"], "person": ["Hiroshi Tanaka"], "product": ["Afeela"], "location": ["Tokyo", "Osaka"]},
     {"company": ["Afeela"], "person": ["Tanaka"], "product": ["concept car"], "location": ["dealers"]},
     "two cities again; possessive 'Sony's' span boundary"),
    ("The iPhone Air, announced by Apple marketing chief Greg Joswiak at the Cupertino event, ships with a "
     "threaded-sensor accessory called FlexCam.",
     {"company": ["Apple"], "person": ["Greg Joswiak"], "product": ["iPhone Air", "FlexCam"], "location": ["Cupertino"]},
     {"company": ["FlexCam"], "person": ["Joswiak"], "product": ["event"], "location": ["accessory"]},
     "TWO products; invented accessory tests span boundary discipline"),
    ("Greta Lindqvist, treasurer at Volvo Cars, presented the electrification roadmap to analysts in Gothenburg, "
     "flanked by the EX90 program leads.",
     {"company": ["Volvo Cars"], "person": ["Greta Lindqvist"], "product": ["EX90"], "location": ["Gothenburg"]},
     {"company": ["EX90"], "person": ["treasurer"], "product": ["roadmap"], "location": ["analysts"]},
     "car model as product; 'program leads' noun-phrase distractor"),
    ("Coinbase rolled out InstantSettle for merchants in Toronto while CFO Alesia Haas fielded questions about "
     "the SEC case.",
     {"company": ["Coinbase"], "person": ["Alesia Haas"], "product": ["InstantSettle"], "location": ["Toronto"]},
     {"company": ["SEC"], "person": ["Haas"], "product": ["questions"], "location": ["merchants"]},
     "'SEC' is an org-like distractor (a regulator, not the operating company here)"),
    ("At the Detroit auto show, Ford's Jim Farley confirmed the F-150 Lightning refresh while Michigan lawmakers "
     "watched from the front row.",
     {"company": ["Ford"], "person": ["Jim Farley"], "product": ["F-150 Lightning"], "location": ["Detroit", "Michigan"]},
     {"company": ["F-150"], "person": ["Farley"], "product": ["refresh"], "location": ["auto show"]},
     "state + city both locations; model name is the product distractor for company"),
    ("Priya Raghavan, product lead at Figma, introduced FigJam AI to the design community at the London "
     "conference, crediting the platform team.",
     {"company": ["Figma"], "person": ["Priya Raghavan"], "product": ["FigJam AI"], "location": ["London"]},
     {"company": ["FigJam AI"], "person": ["Raghavan"], "product": ["conference"], "location": ["design community"]},
     "product name contains the company's brand prefix"),
    ("Toyota's Akio Toyoda personally tested the GR Yaris at the Nurburgring ahead of the Frankfurt dealer "
     "summit.",
     {"company": ["Toyota"], "person": ["Akio Toyoda"], "product": ["GR Yaris"], "location": ["Nurburgring", "Frankfurt"]},
     {"company": ["GR"], "person": ["Toyoda"], "product": ["summit"], "location": ["dealer summit"]},
     "two locations; surname distractor; 'GR' partial product"),
    ("Chime's CFO Michael Rangeloff unveiled ChimeCard Plus for students in Austin, noting the neobank's growth.",
     {"company": ["Chime"], "person": ["Michael Rangeloff"], "product": ["ChimeCard Plus"], "location": ["Austin"]},
     {"company": ["ChimeCard Plus"], "person": ["Rangeloff"], "product": ["neobank"], "location": ["students"]},
     "product embeds the company name; 'neobank' category-word distractor"),
    ("Zara Okafor, the Nigerian fintech founder behind Kuda, pitched the Kuda Overdraft feature to investors in "
     "Lagos.",
     {"company": ["Kuda"], "person": ["Zara Okafor"], "product": ["Kuda Overdraft"], "location": ["Lagos"]},
     {"company": ["Kuda Overdraft"], "person": ["founder"], "product": ["investors"], "location": ["Nigerian"]},
     "founder's nationality is an adjective distractor; product repeats company name"),
    ("Rivian CEO RJ Scaringe showed the R2 crossover at the Laguna Beach design studio, and analysts in New York "
     "replayed the clip all afternoon.",
     {"company": ["Rivian"], "person": ["RJ Scaringe"], "product": ["R2"], "location": ["Laguna Beach", "New York"]},
     {"company": ["R2"], "person": ["Scaringe"], "product": ["clip"], "location": ["design studio"]},
     "two locations; single-letter-ish product 'R2' is a company distractor"),
]
for i, tup in enumerate(_ents, 1):
    text, gt, dis = tup[0], tup[1], tup[2]
    note = tup[3] if len(tup) > 3 else "spans verified as exact substrings; distractors fit a different label"
    TASKS.append({"id": f"ent_{i:02d}", "family": "entities",
                  "split": "tune" if i <= 3 else "test",
                  "input": text, "gt": gt, "distractors": dis,
                  "gt_rationale": note})

# -------------------------------------------------------------- SEVERITY (15)
# gt: level in 5 ordered levels; 3 items per level. Rationale states business impact.
_sev: list[tuple[str, str, str]] = [
    ("The FAQ page shows a 2024 copyright year in the footer.",
     "trivial",
     "cosmetic text only; zero workflow impact"),
    ("A typo in the changelog says 'recieve' instead of 'receive'.",
     "trivial",
     "typo in rarely-read docs; no functional impact"),
    ("The admin panel favicon fails to load on the internal dashboard.",
     "trivial",
     "internal-only cosmetic asset; nobody's work is blocked"),
    ("Email validation rejects addresses with a plus sign, so some users can't sign up until support intervenes.",
     "minor",
     "blocks a subset of signups but support can register them manually"),
    ("The CSV export omits the new 'region' column that power users filter on.",
     "minor",
     "workaround: users re-add the column manually; annoyance not blockage"),
    ("Dark mode renders one settings page with unreadable grey-on-grey text.",
     "minor",
     "one page, one theme; users can switch theme to read it"),
    ("Reported: search doesn't find products with special characters in the name.",
     "moderate",
     "a real functionality gap for a subset of catalog items; workarounds are clumsy"),
    ("The mobile app crashes when rotating the screen during checkout, forcing users to restart the flow.",
     "moderate",
     "loses user progress in a key flow on one platform"),
    ("Invoice PDFs render prices with the wrong currency symbol for European customers.",
     "moderate",
     "customer-facing correctness issue in finance docs; refunds/reissues needed but not mass outages"),
    ("The checkout page throws a 500 whenever a coupon code is applied, so discounted orders cannot complete.",
     "major",
     "blocks a revenue-generating flow for a customer segment; no workaround"),
    ("Single sign-on is failing for all enterprise SSO customers since this morning's deploy.",
     "major",
     "an entire customer segment cannot log in; every affected org is blocked"),
    ("The nightly billing batch has silently skipped 15% of accounts for three nights, under-billing revenue.",
     "major",
     "recurring silent revenue loss compounding nightly; finance impact grows daily"),
    ("A hardcoded credential for the production database was pushed to the public GitHub repo.",
     "critical",
     "live secret exposed publicly: immediate breach risk, incident response required"),
    ("Payments are double-charging some customers during peak hours, and duplicates are refundable only manually.",
     "critical",
     "actively harms customers' money at scale right now; legal/compliance exposure"),
    ("The primary database cluster is unavailable in us-east; all writes are failing for all customers.",
     "critical",
     "total service unavailability for a region: maximum business impact"),
]
for i, (text, level, rat) in enumerate(_sev, 1):
    TASKS.append({"id": f"sev_{i:02d}", "family": "severity",
                  "split": "tune" if i <= 3 else "test",
                  "input": text, "gt": {"level": level, "rationale": rat}, "gt_rationale": rat})

# ---------------------------------------------------------------- VERIFY (15)
# gt: supported yes/no; BALANCED 8 no / 7 yes. Evidence shapes: numeric mismatch, stale date,
# scope mismatch, actually-consistent, consistent-with-confounder.
# Counterfactual logic tested: a fact that COULD explain the mismatch but is itself unverified.
_ver: list[tuple[dict[str, Any], str, str]] = [
    ({"claim": "The invoice total matches the approved purchase order.",
      "evidence": {"invoice": {"number": "INV-88", "total_usd": 2300},
                   "purchase_order": {"id": "PO-112", "approved_total_usd": 1900},
                   "notes": "Finance approved a change order on 2026-08-14 raising the cap to $2,500."}},
     "no", "invoice 2300 != PO approved 1900; change order raises a cap but was not the approved total"),
    ({"claim": "The shipment arrived within the promised 2-day window.",
      "evidence": {"order": {"id": "ORD-501", "placed": "2026-09-01", "promised_delivery": "2026-09-03"},
                   "tracking": {"carrier": "FedEx", "delivered_at": "2026-09-03T16:40:00Z"}}},
     "yes", "delivered 2026-09-03 is within the promised 2026-09-03"),
    ({"claim": "The API's published rate limit is accurate.",
      "evidence": {"docs": {"stated_limit_rpm": 100},
                   "gateway_log": {"measured_limit_rpm": 60, "window": "2026-09-10"}},
                   "notes": "Platform team doubled the limit in the September release notes."},
     "no", "docs say 100 rpm but the gateway measured 60 rpm; a 'September release notes' note is not evidence "
           "the limit changed for this window"),
    ({"claim": "Customer subscription is currently active through year-end.",
      "evidence": {"subscription": {"status": "active", "current_period_end": "2026-12-31"},
                   "billing": {"last_payment": "2026-09-05", "amount_usd": 49},
                   "current_date": "2026-09-18"}},
     "yes", "status active, period ends 2026-12-31, current_date 2026-09-18: consistent and current"),
    ({"claim": "The employee is enrolled in the premium health plan.",
      "evidence": {"hr_record": {"plan": "premium", "enrolled": "2026-01-15"},
                   "payroll_deduction": {"code": "PREM", "active": True},
                   "carrier_confirmation": {"status": "pending", "since": "2026-08-30"}},
     "notes": "HR mentioned verbally that the carrier confirmation was mailed."},
     "no", "carrier confirmation is still 'pending'; a verbal mention in notes is unverified and does not "
           "confirm enrollment"),
    ({"claim": "The deployed version matches the version that passed QA.",
     "evidence": {"qa_signoff": {"version": "2026.9.12", "signed": "2026-09-12"},
                  "deploy_record": {"version": "2026.9.12", "environment": "production", "at": "2026-09-13"}},
     "notes": "QA re-ran the suite on the release candidate build 2026.9.12-rc1 on 2026-09-11."},
     "yes", "QA-signed 2026.9.12 == deployed 2026.9.12; the rc1 re-run predates and doesn't contradict the "
            "final signoff"),
    ({"claim": "The vendor is compliant with our SOC 2 requirement.",
      "evidence": {"vendor": {"name": "DataPipe", "soc2_report": {"status": "expired", "expired_on": "2026-06-30"}},
                   "procurement_policy": {"requires": "current SOC 2 Type II"},
                   "notes": "The vendor emailed that a new report is 'in progress'."}},
     "no", "report is expired as of 2026-06-30; 'in progress' is not a current report; policy requires current"),
    ({"claim": "The reimbursement was paid at the correct per-diem rate.",
      "evidence": {"trip": {"destination": "Berlin", "days": 3, "per_diem_usd": 96},
                   "payout": {"amount_usd": 288, "date": "2026-09-15"}},
      "notes": "Per-diem table row for Germany was updated on 2026-09-01: 96 USD/day."},
     "yes", "3 days x 96 = 288 exactly matches payout; table row is dated and current"),
    ({"claim": "The server outage last week was fully resolved within the 4-hour SLA.",
      "evidence": {"incident": {"id": "INC-2201", "started": "2026-09-11T09:12Z", "resolved": "2026-09-11T14:47Z"},
                   "sla": {"max_hours": 4}},
      "notes": "A status-page banner remained visible for 30 minutes after resolution."},
     "no", "09:12 to 14:47 is 5h35m > 4h SLA; the banner note is a confounder for perception, not duration"),
    ({"claim": "The student's transcript shows junior standing.",
      "evidence": {"transcript": {"standing": "senior", "credits_earned": 92},
                   "catalog": {"junior_min_credits": 60, "senior_min_credits": 90}},
      "notes": "An advisor email mentions a possible standing recalculation next term."},
     "no", "transcript itself says 'senior' (and 92 >= 90 senior threshold); a possible future recalculation "
           "doesn't change the record today"),
    ({"claim": "The refund for the cancelled order has been issued.",
      "evidence": {"order": {"id": "ORD-771", "status": "cancelled", "cancel_date": "2026-09-02"},
                   "payments": {"refund_transactions": []}},
      "notes": "Support created a manual task to look into the refund."},
     "no", "zero refund transactions recorded; a task to 'look into' it is not an issued refund"),
    ({"claim": "The Q3 marketing spend stayed within the approved budget.",
      "evidence": {"budget": {"q3_approved_usd": 40000},
                   "ledger": {"q3_spend_usd": 37900, "pending_invoices_usd": 1900}},
      "notes": "Pending invoices are not yet booked against the quarter."},
     "yes", "booked 37,900 + pending 1,900 = 39,800 <= 40,000 approved; even counting pending committed "
            "spend, the quarter is within budget"),
    ({"claim": "The fleet's charging infrastructure covers all depot locations.",
      "evidence": {"depots": ["A", "B", "C", "D", "E"],
                   "chargers": {"A": 4, "B": 6, "C": 2, "D": 0, "E": 3}},
      "notes": "Depot D plans to install chargers next fiscal year."},
     "no", "depot D has zero chargers; a future plan does not cover it today"),
    ({"claim": "The new pricing applies only to new customers.",
      "evidence": {"pricing_update": {"effective": "2026-10-01", "applies_to": "new_customers_only"},
                   "grandfathering": {"policy": "existing customers keep current rate"}},
      "notes": "Sales flagged one gray-area account still under negotiation."},
     "yes", "update text says new_customers_only and grandfathering policy is explicit; one gray-area account "
            "doesn't contradict the written policy"),
    ({"claim": "The dataset passed the monthly PII scan.",
      "evidence": {"scan": {"scan_id": "SCAN-0918", "date": "2026-09-15", "result": "passed",
                            "rules_version": "2026.3"},
                   "dataset": {"id": "DS-42", "rows": 184000}},
      "notes": "The scan tool's PII rules were updated on 2026-09-16 - after this scan."},
     "yes", "the scan that exists passed on its own rules_version 2026.3; a later rules update doesn't "
            "retroactively fail it"),
]
for i, (inp, supported, rat) in enumerate(_ver, 1):
    inp.pop("eexisting", None)
    TASKS.append({"id": f"ver_{i:02d}", "family": "verify",
                  "split": "tune" if i <= 3 else "test",
                  "input": inp, "gt": {"supported": supported, "rationale": rat}, "gt_rationale": rat})

# ---------------------------------------------------------------- RECORDS (15)
# gt: (buyer,item) purchase pairs from 1-3 events. >=3 items trip GLiNER's pilot failure mode
# (same buyer twice / same item twice / cross-alignment).
_rec: list[tuple[str, list[tuple[str, str]], str]] = [
    ("Nina bought a desk lamp and Omar bought a whiteboard.",
     [("Nina", "desk lamp"), ("Omar", "whiteboard")],
     "two clean events"),
    ("Priya purchased running shoes on Monday and a yoga mat on Friday.",
     [("Priya", "running shoes"), ("Priya", "yoga mat")],
     "SAME BUYER twice - buyer column must repeat"),
    ("The warehouse sold three standing desks to Delta Design and two ergonomic chairs to Delta Design.",
     [("Delta Design", "standing desks"), ("Delta Design", "ergonomic chairs")],
     "company buyer with two purchases; same buyer twice"),
    ("Luca bought oranges and later returned them for a refund.",
     [("Luca", "oranges")],
     "purchase + return of the SAME item; only the buy event is a purchase pair"),
    ("Walmart reported strong back-to-school sales: backpacks led the category, followed by lunch boxes and "
     "notebooks.",
     [],
     "retailer-as-subject sales summary: NO buyer/item purchase events at all"),
    ("Fernando ordered two drone batteries; one arrived defective and was replaced under warranty at no charge.",
     [("Fernando", "drone batteries")],
     "buy + warranty replacement; only one purchase pair"),
    ("Amara bought flour and sugar at the market, then sold her old bread machine to a neighbor.",
     [("Amara", "flour"), ("Amara", "sugar")],
     "two purchases by same buyer + an irrelevant sale event"),
    ("Customer 8841 bought a coffee grinder; customer 8841 also bought descaling solution the same week.",
     [("customer 8841", "coffee grinder"), ("customer 8841", "descaling solution")],
     "numeric buyer id, same buyer twice"),
    ("Acme Corp procured 500 units of widget A and 200 units of widget B from the supplier.",
     [("Acme Corp", "widget A"), ("Acme Corp", "widget B")],
     "company buyer, two items, quantities present"),
    ("Jonas bought a tent, and Mia bought the same model tent the following weekend.",
     [("Jonas", "tent"), ("Mia", "tent")],
     "SAME ITEM bought by two different buyers"),
    ("Leila purchased a bicycle and helmet combo, then upgraded the helmet a week later.",
     [("Leila", "bicycle"), ("Leila", "helmet")],
     "combo purchase + upgrade of one component; same buyer, overlapping item"),
    ("The gift shop's bestselling mug was bought by hundreds of tourists this summer.",
     [],
     "aggregate sales claim: no specific buyer/item event"),
    ("Theo bought cherries at the farmers market and gave half to his sister; she later bought cherries too.",
     [("Theo", "cherries"), ("his sister", "cherries")],
     "same item twice across buyers + a non-purchase gift event"),
    ("Ravi leased a printer from OfficeHub and bought toner for it from OfficeHub as well.",
     [("Ravi", "printer"), ("Ravi", "toner")],
     "mixed acquisition types (lease + purchase) from same seller, same buyer"),
    ("Sofia bought tulips, Marcus bought tulips, and Ingrid bought daffodils.",
     [("Sofia", "tulips"), ("Marcus", "tulips"), ("Ingrid", "daffodils")],
     "three buyers, repeated item across two of them"),
]
# Suite-provided NEGATIVE probes (pre-decided, never invented at runtime): cross-pair and
# role-reversal traps. A model that answers "yes" to everything fails these.
_rec_negatives: dict[str, list[tuple[str, str]]] = {
    "rec_01": [("Nina", "whiteboard"), ("Omar", "desk lamp")],
    "rec_02": [("Priya", "desk lamp")],
    "rec_03": [("Delta Design", "whiteboard")],
    "rec_05": [("warehouse", "standing desks"), ("Delta Design", "backpacks")],
    "rec_07": [("the neighbor", "bread machine"), ("Amara", "bread machine")],
    "rec_09": [("Acme Corp", "widget C")],
    "rec_10": [("Mia", "tent model"), ("Jonas", "the following weekend")],
    "rec_12": [("tourists", "mug"), ("the gift shop", "summer")],
    "rec_13": [("Ingrid", "tulips"), ("Theo", "daffodils")],
    "rec_15": [("Ingrid", "tulips"), ("Sofia", "daffodils")],
}
for i, (text, pairs, rat) in enumerate(_rec, 1):
    rid = f"rec_{i:02d}"
    TASKS.append({"id": rid, "family": "records",
                  "split": "tune" if i <= 3 else "test",
                  "input": text, "gt": {"pairs": pairs, "negative_probes": _rec_negatives.get(rid, [])},
                  "gt_rationale": rat})

# ------------------------------------------------------------------ END SUITE
if __name__ == "__main__":
    from collections import Counter
    fams = Counter(t["family"] for t in TASKS)
    splits = Counter(t["split"] for t in TASKS)
    ver = [t for t in TASKS if t["family"] == "verify"]
    print("total:", len(TASKS))
    print("families:", dict(fams))
    print("splits:", dict(splits))
    print("verify gt:", Counter(t["gt"]["supported"] for t in ver))
    print("severity levels:", Counter(t["gt"]["level"] for t in TASKS if t["family"] == "severity"))
    ctrue = sum(1 for t in TASKS if t["family"] == "conditions" and all(v == "yes" for v in t["gt"].values()))
    cle2 = sum(1 for t in TASKS if t["family"] == "conditions" and sum(1 for v in t["gt"].values() if v == "yes") <= 2)
    print("cond all-5-true:", ctrue, "| cond <=2-true:", cle2)
    print("asp present counts:", Counter(len(t["gt"]) for t in TASKS if t["family"] == "aspects"))
    print("rec pair counts:", Counter(len(t["gt"]["pairs"]) for t in TASKS if t["family"] == "records"))
