# Discovery transcript: FAN-OUT

--- step 0 model ---
Missed small fans — inspecting the cluster to find a low-noise signal.SHOW 123

--- step 0 tool ---
instance 123: 4 members
  2022-09-04 08:04:00 80B7973F0->809BFDCE0 $17,766 [round_trip=0.00,layering_chain=0.00,fan_out=0.00,fan_in=0.00,scatter_gather=0.00,gather_scatter=0.00,structuring_fan_in=0.00,bipartite_butterfly=0.00]
  2022-09-04 11:41:00 80B7973F0->8009C3620 $6,466 [round_trip=0.00,layering_chain=0.00,fan_out=0.00,fan_in=0.20,scatter_gather=0.00,gather_scatter=0.00,structuring_fan_in=0.00,bipartite_butterfly=0.00]
  2022-09-05 09:21:00 80B7973F0->80BE6AE70 $1,882 [round_trip=0.00,layering_chain=0.00,fan_out=0.20,fan_in=0.20,scatter_gather=0.00,gather_scatter=0.00,structuring_fan_in=0.00,bipartite_butterfly=0.00]
  2022-09-07 15:28:00 80B7973F0->80F11E180 $13,017 [round_trip=0.00,layering_chain=0.00,fan_out=0.40,fan_in=0.00,scatter_gather=0.00,gather_scatter=0.00,structuring_fan_in=0.00,bipartite_butterfly=0.00]

--- step 1 model ---
Small counts explain the misses — checking the other instances for a tighter pattern.Thinking small counts score 0 on `many`, need other signal. Checking remaining instances. SHOW 138

--- step 1 tool ---
instance 138: 2 members
  2022-09-04 18:41:00 80E5157A0->8006ECDD0 $15,589 [round_trip=0.00,layering_chain=0.00,fan_out=0.00,fan_in=0.00,scatter_gather=0.00,gather_scatter=0.00,structuring_fan_in=0.00,bipartite_butterfly=0.00]
  2022-09-07 07:35:00 80E5157A0->808CB27A0 $12,040 [round_trip=0.00,layering_chain=0.00,fan_out=0.00,fan_in=0.00,scatter_gather=0.00,gather_scatter=0.00,structuring_fan_in=0.00,bipartite_butterfly=0.00]

--- step 2 model ---
Truth fans vary too much for averages — looking for a cleaner discriminator.SHOW 223

--- step 2 tool ---
instance 223: 1 members
  2022-09-07 00:21:00 812A09CF0->812A09D40 $3,269 [round_trip=0.00,layering_chain=0.00,fan_out=0.00,fan_in=0.00,scatter_gather=0.00,gather_scatter=0.25,structuring_fan_in=0.00,bipartite_butterfly=0.00]

--- step 3 model ---
That single-transfer instance is puzzling — I'll check the last case to map the full picture.Single-member display is odd — checking the last instance before forming a theory. SHOW 224

--- step 3 tool ---
instance 224: 1 members
  2022-09-07 00:53:00 8021353D0->80266F880 $7,206 [round_trip=0.22,layering_chain=0.00,fan_out=0.00,fan_in=0.20,scatter_gather=0.00,gather_scatter=0.38,structuring_fan_in=0.00,bipartite_butterfly=0.35]

--- step 4 model ---
I've confirmed small fan sizes; now I'll test a selective motif that favors small counts to reduce background noise.```motifql
term small_n = trapezoid(1, 2, 4, 6)
motif cand {
  match (s:account) -[e:transfer]-> (r:account)
  group by s over r window 4d
  having count(distinct r) >= 2
  soft count(distinct r) is small_n and each r: e.amount >= 1500 USD
  score min
  emit when score >= 0.5
}
```
TRY

--- step 4 tool ---
no ```motifql block found; send the motif in one.

--- step 5 model ---
Small fans miss on count — trying a tighter, low-noise variant.TRY
```motifql
term small_n = trapezoid(1, 2, 4, 6)
motif cand {
  match (s:account) -[e:transfer]-> (r:account)
  group by s over r window 4d
  having count(distinct r) >= 2
  soft count(distinct r) is small_n
  score min
  emit when score >= 0.5
}
```

--- step 5 tool ---
COMPILED. emit=0.50 cluster recall 4/4 (mean top 1.000)
background fire 10831/20000 (54.155%)
marginal background fire 8651/20000 (43.255%; fires where no current motif fires)
  FP tid=1874397 score=1.00
  FP tid=1876592 score=1.00
  FP tid=1880436 score=1.00

--- step 6 model ---
Count alone is too noisy — testing a stricter shape to bring the background down.Small counts alone fire everywhere — pivoting to tighter scatter shape with low noise.
TRY
```motifql
term several = ramp(1, 3)
motif cand {
  match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)
  group by s, d over m window 7d
  where leg2.time > leg1.time and s != d
  having count(distinct m) >= 2
  soft count(distinct m) is several and sum(leg2.amount) ~= sum(leg1.amount) within 20%
  score product
  emit when score >= 0.5
}
```

--- step 6 tool ---
COMPILED. emit=0.50 cluster recall 0/4 (mean top 0.000)
background fire 0/20000 (0.000%)
marginal background fire 0/20000 (0.000%; fires where no current motif fires)

--- step 7 model ---
Small fans drown in background hubs, so I'm testing amount-selective filters to isolate them.TRY
```motifql
term small_n = trapezoid(1, 2, 4, 6)
term big = ramp(6000 USD, 10000 USD)
motif cand {
  match (s:account) -[e:transfer]-> (r:account)
  group by s over r window 4d
  having count(distinct r) >= 2
  soft count(distinct r) is small_n and most r: e.amount is big
  score product
  emit when score >= 0.5
}
```

--- step 7 tool ---
COMPILED. emit=0.50 cluster recall 3/4 (mean top 0.626)
background fire 729/20000 (3.645%)
marginal background fire 540/20000 (2.700%; fires where no current motif fires)
  FP tid=1872493 score=1.00
  FP tid=1884993 score=0.83
  FP tid=1891329 score=1.00

--- step 8 model ---
Big-amount fans look promising but still noisy – tightening to hit the gate.TRY
```motifql
term small_n = trapezoid(1, 2, 4, 6)
term huge_sum = ramp(20000 USD, 30000 USD)
motif cand {
  match (s:account) -[e:transfer]-> (r:account)
  group by s over r window 4d
  having count(distinct r) >= 2
  soft count(distinct r) is small_n and sum(e.amount) is huge_sum
  score min
  emit when score >= 0.5
}
```

--- step 8 tool ---
COMPILED. emit=0.50 cluster recall 4/4 (mean top 1.000)
background fire 6947/20000 (34.735%)
marginal background fire 5235/20000 (26.175%; fires where no current motif fires)
  FP tid=1874397 score=1.00
  FP tid=1884041 score=1.00
  FP tid=1885352 score=1.00

--- step 9 model ---
Small-fan counts are too common for the background gate — I'm shifting to rarer shared structures.Butterfly background is naturally low — testing if the missed fans participate in shared-receiver cores.
TRY
```motifql
term close_amt = trapezoid(0 USD, 0 USD, 800 USD, 2000 USD)
term tight = trapezoid(0h, 0h, 24h, 72h)
motif cand {
  match (x1:account) -[a:transfer]-> (y1:account), (x1:account) -[b:transfer]-> (y2:account), (x2:account) -[c:transfer]-> (y1:account), (x2:account) -[d:transfer]-> (y2:account)
  where x1 != x2 and y1 != y2 and span(all) <= 7d
  soft span(all) is tight
  score min
  emit when score >= 0.5
}
```

--- step 9 tool ---
COMPILED. emit=0.50 cluster recall 1/4 (mean top 0.202)
background fire 91/20000 (0.455%)
marginal background fire 19/20000 (0.095%; fires where no current motif fires)
  FP tid=1887441 score=1.00
  FP tid=1886268 score=0.50
  FP tid=1877154 score=1.00

--- step 10 model ---
Big-amount pairs look like the right discriminator — tightening to clear the gate.TRY
```motifql
term small_n = trapezoid(1, 2, 4, 6)
motif cand {
  match (s:account) -[e:transfer]-> (r:account)
  group by s over r window 4d
  having count(distinct r) >= 2
  soft count(distinct r) is small_n and at least 2 r: e.amount >= 10000 USD
  score min
  emit when score >= 0.5
}
```

--- step 10 tool ---
COMPILED. emit=0.50 cluster recall 2/4 (mean top 0.500)
background fire 1433/20000 (7.165%)
marginal background fire 839/20000 (4.195%; fires where no current motif fires)
  FP tid=1887818 score=0.50
  FP tid=1877839 score=0.50
  FP tid=1872205 score=1.00
NEXT IS YOUR LAST STEP: reply PROPOSE or DONE.

--- step 11 model ---
No candidate clears both recall and background bars; further proposals would waste the run.
DONE

--- step 11 tool ---
noted.

## Proposal

(none)

## Gate

(no pass)