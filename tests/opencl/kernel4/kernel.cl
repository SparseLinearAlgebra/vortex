#define TS 32
#define WIDTH 4

#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#endif

__kernel void myGEMM4(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global floatX* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS/WIDTH)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = (TS/WIDTH)*get_group_id(0) + row; // Row ID of C (0..M/WIDTH)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local floatX Asub[TS][TS/WIDTH];
    __local floatX Bsub[TS][TS/WIDTH];

    // Initialise the accumulation registers
    #if WIDTH == 1
        floatX acc = 0.0f;
    #elif WIDTH == 2
        floatX acc = { 0.0f, 0.0f };
    #elif WIDTH == 4
        floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    #elif WIDTH == 8
        floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    #endif
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int tile=0; tile<numTiles; tile++) {

        // Load one tile of A and B into local memory
        const int tiledRow = (TS/WIDTH)*tile + row;
        const int tiledCol = TS*tile + col;
        Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];
        Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        floatX vecA, vecB;
        float valB;
        for (int k=0; k<TS/WIDTH; k++) {
            vecB = Bsub[col][k];
            for (int w=0; w<WIDTH; w++) {
                vecA = Asub[WIDTH*k + w][row];
                #if WIDTH == 1
                    valB = vecB;
                    acc += vecA * valB;
                #elif WIDTH == 2
                    switch (w) {
                        case 0: valB = vecB.x; break;
                        case 1: valB = vecB.y; break;
                    }
                    acc.x += vecA.x * valB;
                    acc.y += vecA.y * valB;
                #elif WIDTH == 4
                    switch (w) {
                        case 0: valB = vecB.x; break;
                        case 1: valB = vecB.y; break;
                        case 2: valB = vecB.z; break;
                        case 3: valB = vecB.w; break;
                    }
                    acc.x += vecA.x * valB;
                    acc.y += vecA.y * valB;
                    acc.z += vecA.z * valB;
                    acc.w += vecA.w * valB;
                #elif WIDTH == 8
                    switch (w) {
                        case 0: valB = vecB.s0; break;
                        case 1: valB = vecB.s1; break;
                        case 2: valB = vecB.s2; break;
                        case 3: valB = vecB.s3; break;
                        case 4: valB = vecB.s4; break;
                        case 5: valB = vecB.s5; break;
                        case 6: valB = vecB.s6; break;
                        case 7: valB = vecB.s7; break;
                    }
                    acc.s0 += vecA.s0 * valB;
                    acc.s1 += vecA.s1 * valB;
                    acc.s2 += vecA.s2 * valB;
                    acc.s3 += vecA.s3 * valB;
                    acc.s4 += vecA.s4 * valB;
                    acc.s5 += vecA.s5 * valB;
                    acc.s6 += vecA.s6 * valB;
                    acc.s7 += vecA.s7 * valB;
                #endif
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    C[globalCol*(M/WIDTH) + globalRow] = acc;
}

