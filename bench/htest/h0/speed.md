# sd.htest H0 속도 기준선 (scipy·statsmodels)

scipy 1.18.1, statsmodels 0.15.0, numpy 2.5.4, Python 3.15.0, arm64, 1스레드. 단위는 ms이고 호출 하나의 가장 좋은 시간(최대 5번)이다. 표본은 표준정규(두 번째 표본은 0.1 이동), 배치는 10⁴줄 × 30(axis=-1)이다. 만드는 스크립트는 bench/htest/h0/speed.py, 원자료는 speed.json이다. –는 재지 않은 칸(앞 크기가 5초를 넘었거나 O(n²) 방법)이다.

| 마일스톤 | 검정 | n = 10 | n = 10³ | n = 10⁶ | 배치 10⁴ × 30 |
|---|---|---:|---:|---:|---:|
| H1 | ttest_1samp | 0.136 | 0.135 | 1.63 | 1.96 |
| H1 | ttest_ind | 0.19 | 0.199 | 3.23 | 2.74 |
| H1 | ttest_ind (Welch) | 0.19 | 0.2 | 3.2 |  |
| H1 | ttest_rel | 0.113 | 0.121 | 1.94 | 2.06 |
| H1 | f_oneway (3) | 0.123 | 0.159 | 5.02 | 4.13 |
| H1 | bartlett (3) | 0.176 | 0.17 | 1.97 |  |
| H1 | levene (3) | 0.153 | 0.147 | 23.9 |  |
| H2 | chisquare | 0.0711 | 0.0665 | 0.776 | 0.884 |
| H2 | jarque_bera | 0.137 | 0.148 | 19.6 | 7.49 |
| H2 | normaltest | 0.201 | 0.21 | 19.1 |  |
| H2 | shapiro | 0.0707 | 0.0949 | 74.5 | 3.73 |
| H2 | anderson (interpolate) | 0.0834 | 0.119 | 52.5 |  |
| H2 | ks_1samp (norm) | 0.192 | 0.217 | 79 |  |
| H2 | cramervonmises (norm) | 0.242 | 0.26 | 77.1 |  |
| H3 | mannwhitneyu | 0.152 | 0.202 | 196 | 14.8 |
| H3 | wilcoxon | 0.208 | 0.136 | 91 | 15.7 |
| H3 | ranksums | 0.114 | 0.152 | 179 |  |
| H3 | kruskal (3) | 0.129 | 0.202 | 297 |  |
| H3 | brunnermunzel | 0.183 | 0.252 | 339 |  |
| H3 | ansari | 0.144 | 0.16 | 180 |  |
| H4 | ks_2samp | 0.18 | 0.248 | 162 | 87.5 |
| H4 | cramervonmises_2samp | 2.17 | 0.233 | 147 |  |
| H4 | anderson_ksamp (2) | 0.0299 | 0.217 | – |  |
| H4 | epps_singleton_2samp | 0.354 | 0.434 | 107 |  |
| H5 | pearsonr | 0.0844 | 0.0916 | 2.56 | 1.94 |
| H5 | spearmanrho | 0.198 | 0.256 | 165 | 12.7 |
| H5 | kendalltau | 0.107 | 0.128 | 174 | 950 |
| H5 | weightedtau | 0.0574 | 0.868 | 2,592 |  |
| H5 | chatterjeexi | 0.146 | 0.213 | 310 |  |
| H6 | fisher_exact 2x2 | 0.17 (고정 표) | | | |
| H6 | chi2_contingency 5x5 | 0.0852 (고정 표) | | | |
| H6 | barnard_exact 2x2 | 0.732 (고정 표) | | | |
| H6 | boschloo_exact 2x2 | 1.98 (고정 표) | | | |
| H7 | binomtest (k = 0.45 n) | 0.14 | 0.241 | 0.121 |  |
| H7 | proportions_ztest (2) | 0.0203 | 0.0197 | 0.019 |  |
| H7 | proportion_confint (wilson) | 0.0258 | 0.0257 | 0.027 |  |
| H8 | false_discovery_control (bh) | 0.0312 | 0.0554 | 83.2 | 5.11 |
| H8 | multipletests (holm-sidak) | 0.00863 | 0.0298 | 80.3 |  |
| H8 | multipletests (hommel) | 0.0338 | 4.11 | – |  |
| H8 | combine_pvalues (fisher) | 0.059 | 0.065 | 2.51 |  |
| H8 | tukey_hsd (3) | 18.2 | 28.8 | 2.32 |  |
