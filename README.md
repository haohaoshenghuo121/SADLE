# SADLE

Official model implementation of **Subject-Adaptive Dual-Level Evidential
Fusion of Multi-view Functional Connectivity Networks for Brain Disease
Identification**.

Functional connectivity networks constructed using different connectivity
metrics provide complementary information for brain disease identification.
However, their reliability may vary across subjects and temporal periods.
Existing fusion methods commonly use fixed or globally shared fusion rules,
which cannot adequately account for inter-subject heterogeneity and
intra-subject temporal variability.

SADLE introduces a subject-adaptive dual-level evidential fusion framework.
Resting-state fMRI signals are divided into temporal windows, from which
multiple functional connectivity networks are constructed using Pearson
correlation, high-order functional connectivity, sparse representation, and
mutual information. View-specific graph encoders then learn diagnostic
evidence and its associated uncertainty.

The learned evidence is fused at two levels: metric-view fusion integrates the
complementary evidence within each temporal window, while time-view fusion
combines evidence across windows. By explicitly considering both uncertainty
and conflict during fusion, SADLE produces subject-specific decisions instead
of applying the same fusion strategy to every subject.
