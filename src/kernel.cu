#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1; // Read old velocity here
glm::vec3 *dev_vel2; // Write new velocity updates here -> separation good for parallelism

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

glm::vec3 *dev_pos2;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;


/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_pointer_cast(dev_particleArrayIndices);

  dev_thrust_particleGridIndices = thrust::device_pointer_cast(dev_particleGridIndices);

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos2 failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  glm::vec3 velocityChange(0.0f);

  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  glm::vec3 perceivedCenter(0.0f);
  int numberOfNeighbors1 = 0;

  for (int i = 0; i < N; i++) {
    if (i != iSelf && glm::distance(pos[i], pos[iSelf]) < rule1Distance) {
      numberOfNeighbors1 += 1;
      perceivedCenter += pos[i];
    }
  }

  if (numberOfNeighbors1 > 0) {
    perceivedCenter /= numberOfNeighbors1;
    velocityChange += (perceivedCenter - pos[iSelf]) * rule1Scale;
  }
  

  // Rule 2: boids try to stay a distance d away from each other
  glm::vec3 c(0.0f);

  for (int i = 0; i < N; i++) {
    if (i != iSelf && glm::distance(pos[i], pos[iSelf]) < rule2Distance) {
      c -= (pos[i] - pos[iSelf]);
    }
  }

  velocityChange += c * rule2Scale;

  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 perceived_velocity(0.0f);
  int numberOfNeighbors3 = 0;
  for (int i = 0; i < N; i++) {
    if (i != iSelf && glm::distance(pos[i], pos[iSelf]) < rule3Distance) {
      numberOfNeighbors3 += 1;
      perceived_velocity += vel[i];
    }
  }

  if (numberOfNeighbors3 > 0) {
    perceived_velocity /= numberOfNeighbors3;
    velocityChange += perceived_velocity * rule3Scale;
  }


  return velocityChange;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index >= N) {
    return;
  }
  
  glm::vec3 updatedVelocity = vel1[index] + computeVelocityChange(N, index, pos, vel1);
  
  // Clamp the velocity
  if (glm::length(updatedVelocity) > maxSpeed) {
    updatedVelocity = glm::normalize(updatedVelocity) * maxSpeed;
  }
  
  // Record the new velocity into vel2. Question: why NOT vel1?
  vel2[index] = updatedVelocity;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index >= N) {
      return;
    }

    glm::vec3 currPos = pos[index];

    int gridX = floorf((currPos.x - gridMin.x) * inverseCellWidth);
    int gridY = floorf((currPos.y - gridMin.y) * inverseCellWidth);
    int gridZ = floorf((currPos.z - gridMin.z) * inverseCellWidth);

    int gridIndex = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);
    indices[index] = index; // Boid index row -> Value
    gridIndices[index] = gridIndex; // Grid Index Row -> Key
    
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= N) {
    return;
  }

  int currentGridIndex = particleGridIndices[index];

  if (index == 0) {
    gridCellStartIndices[currentGridIndex] = 0;
  } else {
    int previousGridIndex = particleGridIndices[index - 1];

    if (currentGridIndex != previousGridIndex) {
      gridCellEndIndices[previousGridIndex] = index;
      gridCellStartIndices[currentGridIndex] = index;
    }
  }

  if (index == N - 1) {
    gridCellEndIndices[currentGridIndex] = N;
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index >= N) {
    return;
  }

  int gridX = floorf((pos[index].x - gridMin.x) * inverseCellWidth);
  int gridY = floorf((pos[index].y - gridMin.y) * inverseCellWidth);
  int gridZ = floorf((pos[index].z - gridMin.z) * inverseCellWidth);

  glm::vec3 velocityChange(0.0f);

  // Rule 1: boids fly towards their local perceived center of mass
  glm::vec3 perceivedCenter(0.0f);
  int numberOfNeighbors1 = 0;

  // Rule 2: boids try to stay a distance d away from each other
  glm::vec3 c(0.0f);

  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 perceived_velocity(0.0f);
  int numberOfNeighbors3 = 0;

float cellCenterX = gridMin.x + (gridX + 0.5f) * cellWidth;
float cellCenterY = gridMin.y + (gridY + 0.5f) * cellWidth;
float cellCenterZ = gridMin.z + (gridZ + 0.5f) * cellWidth;

int xDirection = pos[index].x < cellCenterX ? -1 : 1;
int yDirection = pos[index].y < cellCenterY ? -1 : 1;
int zDirection = pos[index].z < cellCenterZ ? -1 : 1;


for (int zChoice = 0; zChoice < 2; zChoice++) {
  for (int yChoice = 0; yChoice < 2; yChoice++) {
    for (int xChoice = 0; xChoice < 2; xChoice++) {
      int neighborGridX = gridX + (xChoice == 0 ? 0 : xDirection);
      int neighborGridY = gridY + (yChoice == 0 ? 0 : yDirection);
      int neighborGridZ = gridZ + (zChoice == 0 ? 0 : zDirection);

      if (neighborGridX < 0 || neighborGridX >= gridResolution ||
          neighborGridY < 0 || neighborGridY >= gridResolution ||
          neighborGridZ < 0 || neighborGridZ >= gridResolution) {
        continue;
      }

      int neighborGridIndex = gridIndex3Dto1D(
        neighborGridX,
        neighborGridY,
        neighborGridZ,
        gridResolution
      );

      int cellStart = gridCellStartIndices[neighborGridIndex];
      int cellEnd = gridCellEndIndices[neighborGridIndex];

      if (cellStart == -1) {
        continue;
      }

      for (int i = cellStart; i < cellEnd; i++) {
        int iOther = particleArrayIndices[i];

        if (iOther == index) {
          continue;
        }

        float distance = glm::distance(pos[iOther], pos[index]);

        if (distance < rule1Distance) {
          numberOfNeighbors1 += 1;
          perceivedCenter += pos[iOther];
        }

        if (distance < rule2Distance) {
          c -= (pos[iOther] - pos[index]);
        }

        if (distance < rule3Distance) {
          numberOfNeighbors3 += 1;
          perceived_velocity += vel1[iOther];
        }
      }
    }
  }
}

  if (numberOfNeighbors1 > 0) {
    perceivedCenter /= numberOfNeighbors1;
    velocityChange +=
      (perceivedCenter - pos[index]) * rule1Scale;
  }

  velocityChange += c * rule2Scale;

  if (numberOfNeighbors3 > 0) {
    perceived_velocity /= numberOfNeighbors3;
    velocityChange += perceived_velocity * rule3Scale;
  }

  glm::vec3 updatedVelocity = vel1[index] + velocityChange;

  if (glm::length(updatedVelocity) > maxSpeed) {
    updatedVelocity = glm::normalize(updatedVelocity) * maxSpeed;
  }

  vel2[index] = updatedVelocity;
}

__global__ void kernReorderParticleData(
  int N, int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel,
  glm::vec3 *posCoherent,
  glm::vec3 *velCoherent) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index >= N) {
    return;
  }

  int boidIndex = particleArrayIndices[index];

  posCoherent[index] = pos[boidIndex];
  velCoherent[index] = vel[boidIndex];
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {

  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index >= N) {
    return;
  }

  int gridX = floorf((pos[index].x - gridMin.x) * inverseCellWidth);
  int gridY = floorf((pos[index].y - gridMin.y) * inverseCellWidth);
  int gridZ = floorf((pos[index].z - gridMin.z) * inverseCellWidth);

  glm::vec3 velocityChange(0.0f);

  glm::vec3 perceivedCenter(0.0f);
  int numberOfNeighbors1 = 0;

  glm::vec3 c(0.0f);

  glm::vec3 perceived_velocity(0.0f);
  int numberOfNeighbors3 = 0;

  float cellCenterX = gridMin.x + (gridX + 0.5f) * cellWidth;
  float cellCenterY = gridMin.y + (gridY + 0.5f) * cellWidth;
  float cellCenterZ = gridMin.z + (gridZ + 0.5f) * cellWidth;

  int xDirection = pos[index].x < cellCenterX ? -1 : 1;
  int yDirection = pos[index].y < cellCenterY ? -1 : 1;
  int zDirection = pos[index].z < cellCenterZ ? -1 : 1;

  for (int zChoice = 0; zChoice < 2; zChoice++) {
    for (int yChoice = 0; yChoice < 2; yChoice++) {
      for (int xChoice = 0; xChoice < 2; xChoice++) {
        int neighborGridX = gridX + (xChoice == 0 ? 0 : xDirection);
        int neighborGridY = gridY + (yChoice == 0 ? 0 : yDirection);
        int neighborGridZ = gridZ + (zChoice == 0 ? 0 : zDirection);

        if (neighborGridX < 0 || neighborGridX >= gridResolution ||
            neighborGridY < 0 || neighborGridY >= gridResolution ||
            neighborGridZ < 0 || neighborGridZ >= gridResolution) {
          continue;
        }

        int neighborGridIndex = gridIndex3Dto1D(
          neighborGridX,
          neighborGridY,
          neighborGridZ,
          gridResolution
        );

        int cellStart = gridCellStartIndices[neighborGridIndex];
        int cellEnd = gridCellEndIndices[neighborGridIndex];

        if (cellStart == -1) {
          continue;
        }

        for (int i = cellStart; i < cellEnd; i++) {
          if (i == index) {
            continue;
          }

          float distance = glm::distance(pos[i], pos[index]);

          if (distance < rule1Distance) {
            numberOfNeighbors1 += 1;
            perceivedCenter += pos[i];
          }

          if (distance < rule2Distance) {
            c -= (pos[i] - pos[index]);
          }

          if (distance < rule3Distance) {
            numberOfNeighbors3 += 1;
            perceived_velocity += vel1[i];
          }
        }
      }
    }
  }

  if (numberOfNeighbors1 > 0) {
    perceivedCenter /= numberOfNeighbors1;
    velocityChange += (perceivedCenter - pos[index]) * rule1Scale;
  }

  velocityChange += c * rule2Scale;

  if (numberOfNeighbors3 > 0) {
    perceived_velocity /= numberOfNeighbors3;
    velocityChange += perceived_velocity * rule3Scale;
  }

  glm::vec3 updatedVelocity = vel1[index] + velocityChange;

  if (glm::length(updatedVelocity) > maxSpeed) {
    updatedVelocity = glm::normalize(updatedVelocity) * maxSpeed;
  }

  vel2[index] = updatedVelocity;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(
      numObjects,
      dev_pos,
      dev_vel1,
      dev_vel2
  );
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(
      numObjects,
      dt,
      dev_pos,
      dev_vel2
  );
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Ping-pong the velocity buffers -> the updated read with newly written velocities after the velocity + pos update round is done!
  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 gridBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    gridSideCount,
    gridMinimum,
    gridInverseCellWidth,
    dev_pos,
    dev_particleArrayIndices,
    dev_particleGridIndices
  );
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  thrust::sort_by_key(
    dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + numObjects,
    dev_thrust_particleArrayIndices
  );
  checkCUDAErrorWithLine("thrust::sort_by_key failed!");


  kernResetIntBuffer<<<gridBlocksPerGrid, blockSize>>>(
    gridCellCount,
    dev_gridCellStartIndices,
    -1
  );

  kernResetIntBuffer<<<gridBlocksPerGrid, blockSize>>>(
    gridCellCount,
    dev_gridCellEndIndices,
    -1
  );
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    dev_particleGridIndices,
    dev_gridCellStartIndices,
    dev_gridCellEndIndices
  );

  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    gridSideCount,
    gridMinimum,
    gridInverseCellWidth,
    gridCellWidth,
    dev_gridCellStartIndices,
    dev_gridCellEndIndices,
    dev_particleArrayIndices,
    dev_pos,
    dev_vel1,
    dev_vel2
  );

  checkCUDAErrorWithLine(
    "kernUpdateVelNeighborSearchScattered failed!"
  );

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    dt,
    dev_pos,
    dev_vel2
  );
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 gridBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);

  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    gridSideCount,
    gridMinimum,
    gridInverseCellWidth,
    dev_pos,
    dev_particleArrayIndices,
    dev_particleGridIndices
  );
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  thrust::sort_by_key(
    dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + numObjects,
    dev_thrust_particleArrayIndices
  );
  checkCUDAErrorWithLine("thrust::sort_by_key failed!");

  kernResetIntBuffer<<<gridBlocksPerGrid, blockSize>>>(
    gridCellCount,
    dev_gridCellStartIndices,
    -1
  );

  kernResetIntBuffer<<<gridBlocksPerGrid, blockSize>>>(
    gridCellCount,
    dev_gridCellEndIndices,
    -1
  );
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    dev_particleGridIndices,
    dev_gridCellStartIndices,
    dev_gridCellEndIndices
  );
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  kernReorderParticleData<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    dev_particleArrayIndices,
    dev_pos,
    dev_vel1,
    dev_pos2,
    dev_vel2
  );
  checkCUDAErrorWithLine("kernReorderParticleData failed!");

  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    gridSideCount,
    gridMinimum,
    gridInverseCellWidth,
    gridCellWidth,
    dev_gridCellStartIndices,
    dev_gridCellEndIndices,
    dev_pos2,
    dev_vel2,
    dev_vel1
  );
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(
    numObjects,
    dt,
    dev_pos2,
    dev_vel1
  );
  checkCUDAErrorWithLine("kernUpdatePos failed!");


  std::swap(dev_pos, dev_pos2);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_pos2);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
