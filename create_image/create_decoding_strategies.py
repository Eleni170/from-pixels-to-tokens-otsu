import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral"
})

words_left = ["nice", "dog", "car", "woman", "guy", "man", "people", "big", "house", "cat"]
probs_left = [0.19, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05]
colors_left = ['#1d70b8'] * 5 + ['black'] * 5

words_right = ["drives", "is", "turns", "stops", "down", "a", "not", "the", "small", "told"]
probs_right = [0.45, 0.25, 0.15, 0.10, 0.03, 0.01, 0.004, 0.003, 0.002, 0.001]
colors_right = ['#1d70b8'] * 3 + ['black'] * 7

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.patch.set_facecolor('white')

plt.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.82, wspace=0.3)

ax1.bar(words_left, probs_left, color='none', edgecolor=colors_left, linewidth=1.5, width=0.5)
ax1.set_xlabel(r'$P(w \mid$' + ' "The")', fontsize=24, labelpad=15)

ax2.bar(words_right, probs_right, color='none', edgecolor=colors_right, linewidth=1.5, width=0.5)
ax2.set_xlabel(r'$P(w \mid$' + ' "The", "car")', fontsize=24, labelpad=15)

for ax in (ax1, ax2):
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, labelleft=False, bottom=False)  # Hide default tick marks

    ax.set_xticklabels(ax.get_xticklabels(), fontstyle='italic', fontsize=12)

ax1.plot([-0.9, -0.9], [0, 1], color='black', linestyle='--', linewidth=1)
ax1.plot([-0.95, -0.85], [1, 1], color='black', linewidth=1)  # Top tick marker
ax1.plot([-0.95, -0.85], [0, 0], color='black', linewidth=1)  # Bottom tick marker
ax1.text(-1.05, 1.0, '1.0', va='center', ha='right', fontsize=16)
ax1.text(-1.05, 0.0, '0.0', va='center', ha='right', fontsize=16)


def draw_bracket_with_text(ax, x_start, x_end, y_base, text_math, text_plain):
    ax.annotate('', xy=(x_start, y_base + 0.04), xytext=(x_start, y_base),
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-0.2", color='#1d70b8', lw=2))
    ax.annotate('', xy=((x_start + x_end) / 2, y_base + 0.04), xytext=(x_start, y_base + 0.04),
                arrowprops=dict(arrowstyle="-", color='#1d70b8', lw=2))
    ax.annotate('', xy=((x_start + x_end) / 2, y_base + 0.05), xytext=((x_start + x_end) / 2, y_base + 0.04),
                arrowprops=dict(arrowstyle="-", color='#1d70b8', lw=2))
    ax.annotate('', xy=(x_end, y_base + 0.04), xytext=((x_start + x_end) / 2, y_base + 0.04),
                arrowprops=dict(arrowstyle="-", color='#1d70b8', lw=2))
    ax.annotate('', xy=(x_end, y_base), xytext=(x_end, y_base + 0.04),
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-0.2", color='#1d70b8', lw=2))

    x_center = (x_start + x_end) / 2
    y_text = y_base + 0.09

    ax.text(x_center, y_text, text_math + text_plain, ha='center', va='bottom', fontsize=20)


draw_bracket_with_text(ax1, x_start=0, x_end=4, y_base=0.7,
                       text_math=r'$\sum_{w \in V_{C1}} P(w \mid$',
                       text_plain=' "The") = 0.65')

draw_bracket_with_text(ax2, x_start=0, x_end=2, y_base=0.7,
                       text_math=r'$\sum_{w \in V_{C1}} P(w \mid$',
                       text_plain=' "The", "car") = 0.85')

fig.suptitle("Otsu Thresholding", fontsize=24, fontweight='bold', y=0.98)
plt.savefig("decoding_strategies.png", dpi=100)
plt.show()
