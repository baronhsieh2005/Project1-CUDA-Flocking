**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Baron Ping-Yeh Hsieh
  * [LinkedIn](https://www.linkedin.com/in/baron-ping-yeh-hsieh-976376290/)
* Tested on: Windows 11, AMD Ryzen 9 7940HS w/ Radeon 780M Graphics, 16GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU

## Demo Screenshot
![](/images/project1_demo_screenshot.png)

## Demo Gif
![](/images/project1_demogif.gif)

## Performance Analysis
To collect the FPS data points, each configuration is allocated a 8 second warm up period, after which 10 one-second FPS samples were collected. The reported value is the average of these samples. The tests were also conducted with VSync turned off and under Release Mode, as specificed in the instructions writeup. 

### How Does Boid # Affect FPS?
For this experiment, the block size for each configurations are all uniformly set at 128 (the given default). 

![](/images/experiment1_estimated_with_visualization.png)

#### With Visualization 

| Boid Count | Uniform Grid (FPS) | Coherent Grid (FPS) | Naive (FPS) |
|-----------:|-------------------:|--------------------:|------------:|
| 1,000      | 1315.41             | 1308.07              | 1304.63      |
| 2,500      | 1234.362             | 1247.76              | 854.98       |
| 5,000      | 1170.49             | 1191.44              | 520.27       |
| 10,000     | 1075.6              | 1155.98              | 250.21       |
| 25,000     | 974.43              | 1091.853              | 53.431        |
| 50,000     | 716.674              | 988.32               | 14.27        |
| 100,000    | 379.37              | 830.65               | 3.61         |


![](/images/experiment1_boid_count_vs_fps.png)
#### Without Visualization 
| Boid Count | Uniform Grid (FPS) | Coherent Grid (FPS) | Naive (FPS) |
|-----------:|-------------------:|--------------------:|------------:|
| 1,000      | 1967.64            | 1951.14             | 1943.63     |
| 2,500      | 1801.18            | 1829.92             | 1093.27     |
| 5,000      | 1682.31            | 1726.02             | 601.496     |
| 10,000     | 1515.76            | 1680.35             | 268.356     |
| 25,000     | 1376.98            | 1623.67             | 54.281      |
| 50,000     | 956.53             | 1510.89             | 15.08       |
| 100,000    | 457.386            | 1326.27             | 3.85        |


- Native Performance looks comparable to its uniform and coherent implementations at first, but the brute force approach scales extremely poorly with larger boid counts and performance collapses (hard to imagine the fps if 3 more 0s are added to the boid count... someone could do that test tho! My old computer would not survive this.)
- Uniform scales much better as it limits neighbor checks to nearby cells (instead of brute force all cells). However, at high boid counts there is still a pretty noticeable drop of fps (around 50000 or so).
- Coherent scales the best out of the three implementations, even at 100,000 boid counts there still isn't a noticeable drop (<200 fps dip), showing that reordering boid data improves memory locality enough to offset the extra preprocessing cost contrasted to uniform grid.
- Overall, by increasing the number of boids being simulated, FPS would decrease. This is because more boids means the GPU has more particles to update and more neighbor interactions to check each frame. Since each frame requires more work, fewer frames can be completed per second, so FPS decreases.


### How Does Block Size Affect FPS?
For this experiment, the boid count for each configurations are all uniformly set at 50000, tested using the Coherent Grid implementation.
![](/images/experiment2_block_size_vs_fps.png) 

| Block Size | FPS |
|-----------:|----:|
| 32         | 1318.48 |
| 64         | 1454.54 |
| 128        | 1510.89 |
| 256        | 1472.69 |
| 512        | 1439.33 |

- Performance improves as the block size increases from 32 to 128 threads, with 128 threads per block giving the best result at 1510.89 FPS. Past that point, performance drops slightly at 256 and 512 threads. My assumption to why this happens is because once the block size is large enough, the benefit starts to level off, and even larger blocks can reduce performance because fewer blocks are available to schedule at once, which can lower overall flexibility and occupancy.


### For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

- For this question, I think in the case of the data I collected, the coherent uniform grid generally performed better than the regular uniform grid, and the advantage became much larger as the boid count increased. At 1,000 boids there was almost no difference, but by 50,000 boids the coherent version reached about 1511 FPS compared with 957 FPS, and at 100,000 boids it was about 1326 FPS compared with 457 FPS. It is indeed the outcome that I expected since under the coherent implementation, boids are reordered such that nearby boids are stored closer together in memory. So when there are more boids and each frame involves many more memory accesses, because nearby boids are stored together, so those accesses are more cache-friendly and efficient and thus doesn't nuke the frame rate as much (L bozo for Uniform on the other hand).


### Did changing cell width and checking 27 vs 8 neighboring cells affect performance? 
For this experiment, the boid count for each configurations are all uniformly set at 50000 and the block size at 128.

![](/images/experiment3_8cell_vs_27cell.png) 

| Grid Method | 8-Cell (FPS) | 27-Cell (FPS) |
|-------------|-------------:|--------------:|
| Uniform Grid | 956.53      | 1020.80       |
| Coherent Grid | 1510.89    | 1568.89       |

- It seems like for both Uniform and Coherent Grid implementations, checking 27 cells are both slightly faster than the checking 8 cells configurations. It's a bit counter-intuitive (since we should be checking more grid cells under the 27 config), but I would guess that the reason this is the case is that the smaller cell width means each cell contains fewer boids, so fewer unnecessary boid-to-boid distance checks are performed overall, and thus that reduction in neighbor work was enough to outweigh the extra cell lookups.