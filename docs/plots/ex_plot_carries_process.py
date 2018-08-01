from exotic_dpps import *
import matplotlib.pyplot as plt

N, b = 100, 10 # Size / basis

### Carries process
Carries = carries_process(N, b)

### IID Bernoullis
p = 0.5*(1-1/b) # Bernoullis' parameter
seg_0N = np.arange(0, N) # Segment [0, N]
Ber = seg_0N[np.random.rand(N) < p]


# Display Carries and Bernoullis
fig, ax = plt.subplots(figsize=(17,2))

ax.scatter(Carries, np.ones_like(Carries), color='r', s=20, label='Carries')
ax.scatter(Ber, -np.ones_like(Ber), color='b', s=20, label='Bernoulli')

# Spine options
ax.spines['bottom'].set_position('center')
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Ticks options
minor_ticks = np.arange(0,N+1)                                            
major_ticks = np.arange(0, N+1, 20)                                               
ax.set_xticks(major_ticks)                                                       
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels(major_ticks, fontsize=15)
ax.xaxis.set_ticks_position('bottom')

ax.tick_params(
    axis='y',				# changes apply to the y-axis
    which='both',		# both major and minor ticks are affected
    left=False,			# ticks along the left edge are off
    right=False,		# ticks along the right edge are off
    labelleft=False)# labels along the left edge are off

ax.xaxis.grid(True)
ax.set_xlim([-1,101])
ax.legend(bbox_to_anchor=(0,0.85), frameon=False, prop={'size':20})

plt.show()