/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


var dl = deeplearn;
var Array3D = dl.Array3D;
var InputProvider = dl.InputProvider;
var Tensor = dl.Tensor;
var Optimizer = dl.Optimizer;
var CostReduction = dl.CostReduction;
var FeedEntry = dl.FeedEntry;
var Session = dl.Session;
var NDArrayMath = dl.NDArrayMath;
var NDArray = dl.NDArray;
var Scalar = dl.Scalar;
var GraphRunnerEventObserver = dl.GraphRunnerEventObserver;


const DEFAULT_EVAL_INTERVAL_MS = 1500;
const DEFAULT_COST_INTERVAL_MS = 15000;
const DEFAULT_TEST_COST_INTERVAL_MS = 40000;
const DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS = 3000;


var MetricReduction = {
    SUM: 0,
    MEAN: 1
}

/**
 * A class that drives the training of a graph model given a dataset. It allows
 * the user to provide a set of callbacks for measurements like cost, accuracy,
 * and speed of training.
 */
class MyGraphRunner {

    constructor(math, eventObserver) {
        this.math = math;
        // this.session = session;
        this.eventObserver = eventObserver;

        this.discCostTensor = null;
        this.genCostTensor = null;
        this.discTrainFeedEntrie = null;
        this.genTrainFeedEntries = null;
        this.batchSize = null;
        this.genOptimizer = null;
        this.discOptimizer = null;
        this.currentTrainLoopNumBatches = null;
        this.costIntervalMs = null;

        this.genImageTensor = null;
        this.discPredictionFakeTensor = null;
        this.discPredictionRealTensor = null;
        this.inferenceFeedEntries = null;
        this.inferenceExampleIntervalMs = null;
        this.inferenceExampleCount = null;

        this.testExampleCount = 0;

        // Runtime information=null.
        this.isTraining = null;
        this.totalBatchesTrained = null;
        this.batchesTrainedThisRun = null;
        this.lastComputedMetric = null;

        this.isInferring = null;
        this.lastInferTimeoutID = null;
        this.currentInferenceLoopNumPasses = null;
        this.inferencePassesThisRun = null;

        this.isEvaluating = null;
        this.totalBatchesEvaluated = 0;
        // this.totalTestImagesEvaluated = 0;
        this.batchesEvaluatedThisRun = null;

        this.trainStartTimestamp = null;
        this.evalStartTimestamp = null;
        this.lastCostTimestamp = 0;
        this.lastEvalTimestamp = 0;
        this.lastTestCostTimestamp = 0;

        this.lastStopTimestamp = null;
        this.totalIdleTimeMs = 0;

        this.zeroScalar = null;
        this.metricBatchSizeScalar = null;

        this.resetStatistics();
        this.zeroScalar = Scalar.new(0);
    }

    resetStatistics() {
        this.totalBatchesTrained = 0;
        this.totalBatchesEvaluated = 0;
        // this.totalTestImagesEvaluated = 0;
        this.totalIdleTimeMs = 0;
        this.lastStopTimestamp = null;
    }

    evaluate(critCostTensor, testExampleCount, critTrainFeedEntries,
        critTestFeedEntries, batchSize, critOptimizer,
        genOptimizer, numBatches = null,
        costIntervalMs = DEFAULT_COST_INTERVAL_MS, testCostIntervalMs = DEFAULT_TEST_COST_INTERVAL_MS) {
        this.critCostTensor = critCostTensor;
        // this.genCostTensor = genCostTensor;
        this.critTrainFeedEntries = critTrainFeedEntries;
        this.critTestFeedEntries = critTestFeedEntries;
        this.batchSize = batchSize;
        this.testExampleCount = testExampleCount;
        this.critOptimizer = critOptimizer;
        // this.genOptimizer = genOptimizer;

        this.costIntervalMs = costIntervalMs;
        this.testCostIntervalMs = testCostIntervalMs;
        this.currentTrainLoopNumBatches = numBatches;

        this.batchesEvaluatedThisRun = 0;
        this.isEvaluating = true;
        this.evalStartTimestamp = performance.now();
        this.evaluateNetwork();
    }

    stopEvaluating() {
        this.isEvaluating = false;
        this.lastStopTimestamp = performance.now();
    }

    resumeEvaluating() {
        this.isEvaluating = true;
        if (this.lastStopTimestamp != null) {
            this.totalIdleTimeMs += performance.now() - this.lastStopTimestamp;
        }
        this.evaluateNetwork();
    }

    evaluateNetwork() {
        if (this.batchesEvaluatedThisRun === this.currentTrainLoopNumBatches) {
            this.stopEvaluating();
        }

        if (!this.isEvaluating) {
            if (this.eventObserver.doneEvaluatingCallback != null) {
                this.eventObserver.doneEvaluatingCallback();
            }
            return;
        }

        const start = performance.now();
        const shouldComputeCost = (this.eventObserver.critCostCallback != null) &&
            (start - this.lastCostTimestamp > this.costIntervalMs);
        if (shouldComputeCost) {
            this.lastCostTimestamp = start;
        }

        const shouldComputeTestCost = (this.eventObserver.critCostCallback != null) &&
            (start - this.lastTestCostTimestamp > this.testCostIntervalMs);
        if (shouldComputeTestCost) {
            this.lastTestCostTimestamp = start;
        }

        const costReduction =
            shouldComputeCost ? CostReduction.MEAN : CostReduction.NONE;

        this.math.scope((keep, track) => {
            const critCost = this.session.train(
                this.critCostTensor, this.critTrainFeedEntries, this.batchSize,
                this.critOptimizer, costReduction);

            // const genCost = this.session.train(
            //     this.genCostTensor, this.genTrainFeedEntries, this.batchSize,
            //     this.genOptimizer, costReduction);

            if (shouldComputeCost) {
                const evalTime = performance.now() - start;

                this.eventObserver.critCostCallback(critCost, -1);
                // this.eventObserver.genCostCallback(genCost);

                if (this.eventObserver.evalExamplesPerSecCallback != null) {
                    const evalExamplesPerSec = (this.batchSize * 1000 / evalTime);
                    this.eventObserver.evalExamplesPerSecCallback(evalExamplesPerSec);
                }
            }


            if (shouldComputeTestCost) {

                let critTestCost = this.zeroScalar;

                for (let i = 0; i < this.testExampleCount; i++) {
                    const ndarrayFeedEntries = [];

                    for (let j = 0; j < this.critTestFeedEntries.length; j++) {
                        const feedEntry = this.critTestFeedEntries[j];
                        const nextCopy = (feedEntry.data).getNextCopy(this.math);
                        ndarrayFeedEntries.push({
                            tensor: feedEntry.tensor,
                            data: track(nextCopy)
                        });
                    }

                    critTestCost = this.math.add(critTestCost, this.session.eval(this.critCostTensor, ndarrayFeedEntries));
                }

                const testExampleCountScalar = Scalar.new(this.testExampleCount);

                critTestCost = this.math.divide(critTestCost, testExampleCountScalar);

                // this.totalTestImagesEvaluated += this.testExampleCount

                this.eventObserver.critCostCallback(-1, critTestCost);

            }

            if (this.eventObserver.evalTotalTimeCallback != null) {
                this.eventObserver.evalTotalTimeCallback(
                    (start - this.evalStartTimestamp) / 1000);
            }

            this.batchesEvaluatedThisRun++;
            this.totalBatchesEvaluated++;

            if (this.eventObserver.batchesEvaluatedCallback != null) {
                this.eventObserver.batchesEvaluatedCallback(this.totalBatchesEvaluated);
            }

        });
        requestAnimationFrame(() => this.evaluateNetwork());
    }


    infer(
        genImageTensor, discPredictionFakeTensor,
        discPredictionRealTensor, inferenceFeedEntries,
        inferenceExampleIntervalMs = DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS,
        inferenceExampleCount = 5, numPasses = null) {
        if (this.eventObserver.inferenceExamplesCallback == null &&
            this.eventObserver.inferenceExamplesPerSecCallback == null) {
            throw new Error(
                'Cannot start inference loop, no inference example or ' +
                'examples/sec observer provided.');
        }

        // Make sure the feed values are providers, and not NDArrays.
        for (let i = 0; i < inferenceFeedEntries.length; i++) {
            const feedEntry = inferenceFeedEntries[i];

            if (feedEntry.data instanceof NDArray) {
                throw new Error(
                    'Cannot start inference on the model runner with feed entries of ' +
                    'type NDArray. Please use InputProviders.');
            }
        }

        this.inferenceExampleIntervalMs = inferenceExampleIntervalMs;
        this.genImageTensor = genImageTensor;
        this.discPredictionFakeTensor = discPredictionFakeTensor;
        this.discPredictionRealTensor = discPredictionRealTensor;
        this.inferenceFeedEntries = inferenceFeedEntries;
        this.inferenceExampleCount = inferenceExampleCount;
        this.currentInferenceLoopNumPasses = numPasses;
        if (!this.isInferring) {
            this.inferencePassesThisRun = 0;
            requestAnimationFrame(() => this.inferNetwork());
        }
        this.isInferring = true;
    }

    inferNetwork() {
        if (!this.isInferring ||
            this.inferencePassesThisRun === this.currentInferenceLoopNumPasses) {
            return;
        }

        this.math.scope((keep, track) => {
            // const feeds = [];
            const genImageValues = [];
            // const discPredictionFakeValues = [];
            // const discPredictionRealValues = [];

            const start = performance.now();
            for (let i = 0; i < this.inferenceExampleCount; i++) {
                // Populate a new FeedEntry[] populated with NDArrays.
                const ndarrayFeedEntries = [];
                const ndarrayFeedEntriesCopy = [];


                // for (let j = 0; j < this.inferenceFeedEntries.length; j++) {
                let j = 1;
                const feedEntry = this.inferenceFeedEntries[j];
                const nextData = track((feedEntry.data).getNextCopy(this.math));
                const dataCopy = track((NDArray.like(nextData)));
                // ndarrayFeedEntries.push({
                //     tensor: feedEntry.tensor,
                //     data: nextData
                // });
                ndarrayFeedEntriesCopy.push({
                    tensor: feedEntry.tensor,
                    data: dataCopy
                });
                // }
                // feeds.push(ndarrayFeedEntries);

                const evaluatedTensors = this.session.evalAll(
                    [this.genImageTensor],
                    ndarrayFeedEntriesCopy
                );

                genImageValues.push(track(NDArray.like(evaluatedTensors[0])));
                // discPredictionFakeValues.push(track(NDArray.like(evaluatedTensors[1])));
                // discPredictionRealValues.push(track(NDArray.like(evaluatedTensors[2])));
            }

            if (this.eventObserver.inferenceExamplesPerSecCallback != null) {
                // Force a GPU download, since inference results are generally needed on
                // the CPU and it's more fair to include blocking on the GPU to complete
                // its work for the inference measurement.

                const inferenceExamplesPerSecTime = performance.now() - start;

                const examplesPerSec =
                    (this.inferenceExampleCount * 1000 / inferenceExamplesPerSecTime);
                this.eventObserver.inferenceExamplesPerSecCallback(examplesPerSec);
            }

            if (this.eventObserver.inferenceExamplesCallback != null) {
                this.eventObserver.inferenceExamplesCallback(
                    [], genImageValues
                    // , discPredictionFakeValues, discPredictionRealValues

                );
            }
            this.inferencePassesThisRun++;

        });
        this.lastInferTimeoutID = window.setTimeout(
            () => this.inferNetwork(), this.inferenceExampleIntervalMs);
    }

    stopInferring() {
        this.isInferring = false;
        window.clearTimeout(this.lastInferTimeoutID);
    }

    isInferenceRunning() {
        return this.isInferring;
    }

    getTotalBatchesTrained() {
        return this.totalBatchesTrained;
    }

    getTotalBatchesEvaluated() {
        return this.totalBatchesEvaluated;
    }

    getLastComputedMetric() {
        return this.lastComputedMetric;
    }

    setMath(math) {
        this.math = math;
    }

    setSession(session) {
        this.session = session;
    }

    setInferenceExampleCount(inferenceExampleCount) {
        this.inferenceExampleCount = inferenceExampleCount;
    }
}