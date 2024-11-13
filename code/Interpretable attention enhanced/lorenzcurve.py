import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

road_clipped = gpd.read_file(r'post_processing\data\connected_clip0410.shp')

road_clipped_len = road_clipped['length'].sort_values()

cumulative_length = np.cumsum(road_clipped_len) / np.sum(road_clipped_len)
cumulative_index = np.arange(1, len(road_clipped_len) + 1) / len(road_clipped_len)

cumulative_length = np.insert(cumulative_length, 0, 0)
cumulative_index = np.insert(cumulative_index, 0, 0)

plt.figure(figsize=(8, 8))

ax = plt.gca()  
ax.set_facecolor('white') 

ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, zorder=0)

plt.plot(cumulative_index, cumulative_length, linestyle='-', color='red', zorder=5, label='Lorenz Curve')
lorenz_line, = plt.plot([0, 1], [0, 1], linestyle='--', color='gray', zorder=5)  # 完全平等线

slope = (cumulative_length[-1] - cumulative_length[-2]) / (cumulative_index[-1] - cumulative_index[-2])
intercept = cumulative_length[-1] - slope * cumulative_index[-1]
F_star = -intercept / slope

tangent_line, = plt.plot([F_star, 1], [0, cumulative_length[-1]], linestyle='--', color='green', zorder=5, label='Tangent Line')  # 切线

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xlabel("Street Index", fontsize=12) 
plt.ylabel("Cumulative Length Proportion", fontsize=12) 
plt.title('Cumulative Lorenz Curve')

plt.text(F_star, 0, f'F*={F_star:.3f}', color='green', fontsize=12, ha='center', va='bottom')


ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

for spine in ax.spines.values():
    spine.set_edgecolor('gray')

plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
# plt.show()
# ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------- #


road_clipped_len = road_clipped['length'].sort_values(ascending=False)
road_clipped_len.plot(kind='hist',density=True,bins=100, alpha=0.8,edgecolor='black', color='lightblue',ax=ax)

threshold = road_clipped_len[int(len(road_clipped_len)*F_star)]
# ---------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------- #


######################################   pow law   ###########################################
bins = 100
box_wide = (int(road_clipped_len.max()) - int(road_clipped_len.min())) / bins
box_range = [[round(box_wide * (i + 1), 2), round(box_wide * (i + 2), 2)] for i in range(bins)]
box_range.insert(0, [0, box_range[0][0]])
road_clipped_len = pd.DataFrame(road_clipped_len)

dict_save = {}

for boxes in box_range:
    a = 0
    dict_save[round(np.mean(boxes), 2)] = a
    for index, item in road_clipped_len.iterrows():
        if boxes[0] < item[0] < boxes[1]:
            dict_save[round(np.mean(boxes), 2)] += 1

x_values = list(dict_save.keys())
y_values = list(dict_save.values())

plt.figure(figsize=(8, 6))


ax = plt.gca()
ax.set_facecolor('white')

plt.loglog(x_values, y_values, 'o', color='gray', linestyle='None', markersize=2, label='Data points')

ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5)

log_x = np.log(x_values)
log_y = np.log(y_values)
non_inf_indices = log_y != -np.inf 

log_x = log_x[non_inf_indices]
log_y = log_y[non_inf_indices]

coefficients = np.polyfit(log_x, log_y, 1)
fit_function = np.poly1d(coefficients)

fit_y = np.exp(fit_function(np.log(x_values)))
plt.loglog(x_values, fit_y, 'r-', label='Fit line') 

slope = coefficients[0]
intercept = coefficients[1]

plt.text(0.05, 0.9, f'Y = {slope:.2f} X + {intercept:.2f}', 
         transform=ax.transAxes, color='gray', fontsize=12)

# 
plt.xlabel('Street Length (log scale)')
plt.ylabel('Frequency (log scale)')
# plt.title('Data points with logarithmic fit')
plt.legend()

# 
plt.tight_layout()
plt.show()


