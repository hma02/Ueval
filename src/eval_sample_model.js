// TODOs:
// 1. made critic Initialization same by setting random seed, but can't make input provider Math.random() with a seed
// 2. improve UI description
// 3. upload images to evaluate (https://stackoverflow.com/questions/22087076/how-to-make-a-simple-image-upload-using-javascript-html)
// 4. Get better initial gen from GAN demo

class Net { // gen or disc or critic
    constructor(name, archType, modelConfigs) {
        this.name = name;
        this.archType = archType;
        this.path = modelConfigs[name].paths[this.archType];
        this.inputShape = modelConfigs[name].inputShape;
        this.outputShape = modelConfigs[name].outputShape;
        this.isValid = false;
        this.hiddenLayers = [];
    }

    addLayer() {

        const modelLayer = new ModelLayer(); //document.createElement('model-layer');

        const lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1];

        const lastOutputShape = lastHiddenLayer != null ? lastHiddenLayer.getOutputShape() : this.inputShape;
        this.hiddenLayers.push(modelLayer);

        modelLayer.initialize(window, lastOutputShape);
        // layerParamChanged(which)

        return modelLayer;
    }

    // layerParamChanged() {
    //     // Go through each of the model layers and propagate shapes.

    //     let lastOutputShape = this.inputShape;

    //     for (let i = 0; i < this.hiddenLayers.length; i++) {
    //         lastOutputShape = this.hiddenLayers[i].setInputShape(lastOutputShape);
    //     }
    // }

    validateNet() {
        let valid = true;

        var HiddenLayers = this.hiddenLayers;
        var lastLayerOutputShape = this.outputShape;

        for (let i = 0; i < HiddenLayers.length; ++i) {
            valid = valid && HiddenLayers[i].isValid();
        }
        if (HiddenLayers.length > 0) {
            const lastLayer = HiddenLayers[HiddenLayers.length - 1];
            valid = valid &&
                util.arraysEqual(lastLayerOutputShape, lastLayer.getOutputShape()); // for gen ,  lastLayerOutputShape = inputShape, for critic, lastLayerOutputShape = labelShape
        }

        this.isValid = valid && (HiddenLayers.length > 0);
    }
}

function buildImageContainer(inferenceContainer, model) {

    inferenceContainer.innerHTML = '';
    inferenceContainer.style.lineHeight = "1";
    inferenceContainer.style.minWidth = '155px';
    inferenceContainer.style.minHeight = '155px';

    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {

        if (i % INFERENCE_EXAMPLE_ROWS === 0 && i !== 0) {
            linebreak = document.createElement("br");
            inferenceContainer.appendChild(linebreak);
        }

        const inferenceExampleElement = document.createElement('div');
        inferenceExampleElement.className = 'inference-example';
        inferenceExampleElement.style.display = 'inline';

        // Set up the input visualizer.
        const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement)

        ndarrayImageVisualizer.setShape(model.criticNet.inputShape);
        ndarrayImageVisualizer.setSize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        model.ndarrayVisualizers.push(ndarrayImageVisualizer);

        inferenceContainer.appendChild(inferenceExampleElement);
    }
}


function smoothExamplesPerSec(
    lastExamplesPerSec, nextExamplesPerSec) {
    return Number((EXAMPLE_SEC_STAT_SMOOTHING_FACTOR * lastExamplesPerSec +
            (1 - EXAMPLE_SEC_STAT_SMOOTHING_FACTOR) * nextExamplesPerSec)
        .toPrecision(3));
}

// refactor this load functions into generic utility functions by separating  
// global variables outside and using callback for async return

function loadNetFromJson(modelJson, which) {
    var lastOutputShape;
    var hiddenLayers;

    lastOutputShape = which.inputShape;

    hiddenLayers = which.hiddenLayers;

    const layerBuilders = JSON.parse(modelJson);
    for (let i = 0; i < layerBuilders.length; i++) {
        const modelLayer = which.addLayer();
        modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);

    }
}

class EvalSampleModel {

    constructor(configs, id, needGen = true, sampleImage = false) {
        this.configs = configs;
        this.modelConfigs = this.configs.modelConfigs;

        this.id = id; // prepare for scaling up to multiple models

        this.needGen = needGen;
        this.sampleImage = sampleImage;

        if (this.needGen) {
            this.generatorNet = new Net('gen', 'Convolutional', this.modelConfigs);
        } else {
            if (this.sampleImage == true) {
                this.sampleImageDataLoaded = false;
                this.sampleImageDataSet = null;
            }
        }

        this.criticNet = new Net('crit', 'Convolutional', this.modelConfigs);

        this.metricName = METRIC_NAME;

        const eventObserver = {
            batchesEvaluatedCallback: (batchesEvaluated) =>
                this.displayBatchesEvaluated(batchesEvaluated),

            critCostCallback: (cost) => {
                var batchesEvaluated = this.graphRunner.getTotalBatchesEvaluated();
                this.displayCost(cost, batchesEvaluated)
            },

            inferenceExamplesCallback:
                (inputFeeds, inferenceOutputs) =>
                this.displayInferenceExamplesOutput(inputFeeds, inferenceOutputs),

            evalExamplesPerSecCallback: (examplesPerSec) =>
                this.displayEvalExamplesPerSec(examplesPerSec),
        };

        this.graphRunner = new MyGraphRunner(eventObserver);

        updateSelectedEnvironment(selectedEnvName, this.graphRunner);

        this.isValid = false;
        this.modelInitialized = false;

    }

    initialize(genWeightsPath = null) {

        this.evalBaseElt = document.getElementById(`eval${this.id}`);

        this.loadNetFromPath(this.criticNet.path, this.criticNet);
        if (this.needGen) {
            this.loadNetFromPath(this.generatorNet.path, this.generatorNet);

            this.genWeightsloaded = false;

            if (genWeightsPath != null) {
                this.loadGenWeightsFromPath(genWeightsPath);
            }

            document.getElementById('title' + `${this.id}`).innerText = 'Generator of a GAN';
            document.getElementById('in_subtitle' + `${this.id}`).innerText = 'Generated Data Distribution:';
            document.getElementById('out_subtitle' + `${this.id}`).innerText = 'Generated Data Eval Result:';
        } else if (this.sampleImage) {
            document.getElementById('title' + `${this.id}`).innerText = 'Uploaded Sample';
            document.getElementById('in_subtitle' + `${this.id}`).innerText = 'Sample Data Distribution:';
            document.getElementById('out_subtitle' + `${this.id}`).innerText = 'Sample Data Eval Result:';
        }

        // image visualizers
        this.ndarrayVisualizers = [];
        this.visualizerElt = document.querySelector('#image-container' + `${this.id}`);
        buildImageContainer(this.visualizerElt, this);

        // loss chart
        this.chartData = [];
        this.chartElt = document.getElementById('losschart' + `${this.id}`);
        this.chartElt.style.minWidth = '155px';

        this.critLossChart = createChart(this.chartElt, this.metricName, this.chartData, 0, this.chartData.y);
        this.critLossChart.update();

        // batchesEvaluated
        this.batchesEvaluatedElt = document.getElementById("examplesEvaluated" + `${this.id}`);

        // progress bar
        this.evalProgressBar = document.getElementById("evalBar" + `${this.id}`);
        this.targetEvalExampleAmount = TARGET_EVALUATION_EXAMPLE_AMOUNT;
        this.evalProgressBarBackground = document.getElementById("evalBarBackground" + `${this.id}`);
        this.evalProgressBarBackground.style.visibility = 'hidden';

        // final score
        this.finalScoreElt = document.getElementById("final-score" + `${this.id}`);

        // examples per sec
        this.evalExamplesPerSec = 0;
        this.examplesPerSecElt = document.getElementById("evalExamplesPerSec" + `${this.id}`);

        this.eval_request = null;
        this.btn_eval = document.getElementById('buttoneval' + `${this.id}`);
        this.eval_paused = true;
        this.btn_eval.addEventListener('click', () => {
            this.eval_paused = !this.eval_paused;

            if (this.eval_paused) {
                this.stopEvaluating(); // can return quickly
            } else {
                this.eval_request = true;
            }
        });
    }

    loadNetFromPath(modelPath, which) {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', modelPath);

        xhr.onload = () => {
            loadNetFromJson(xhr.responseText, which);
            // which.layerParamChanged()
            which.validateNet();

            if (this.needGen) {
                this.isValid = this.criticNet.isValid && this.generatorNet.isValid;
            } else {
                this.isValid = this.criticNet.isValid
            }

            console.log('model ID:', this.id, `${which.name}valid`, which.isValid, 'allvalid:', this.isValid, 'genloaded:', this.genWeightsloaded);

            if (this.needGen) {
                if (this.isValid && this.genWeightsloaded) {
                    this.createModel(this.loadedWeights);
                }
            } else {
                if (this.isValid) {
                    this.createModel();
                }
            }

        };
        xhr.onerror = (error) => {
            throw new Error(
                'Model could not be fetched from ' + modelPath + ': ' + error);
        };
        xhr.send();
    }

    loadGenWeightsFromPath(genWeightsPath) {

        const _xhr = new XMLHttpRequest();
        _xhr.open('GET', genWeightsPath);
        _xhr.onload = () => {
            var weightsJson = _xhr.responseText;
            this.loadedWeights = JSON.parse(weightsJson);

            this.genWeightsloaded = true

            console.log('model ID:', this.id, 'allvalid:', this.isValid, 'genloaded:', this.genWeightsloaded);
            if (this.needGen) {
                if (this.isValid && this.genWeightsloaded) {
                    this.createModel(this.loadedWeights);
                }
            } else {
                if (this.isValid) {
                    this.createModel();
                }
            }

        }
        _xhr.onerror = (error) => {
            throw new Error(
                'Model could not be fetched from ' + genWeightsPath + ': ' + error);
        };
        _xhr.send();

    }

    displayCost(avgCost, batchesEvaluated) {

        var cost = avgCost.get();

        this.finalScoreElt.innerHTML = `${this.metricName} Eval Score: ${cost.toPrecision(5)}`;

        this.chartData.push({
            x: batchesEvaluated,
            y: cost
        });
        config.data.datasets[0].data = this.chartData;
        this.critLossChart.update();

    }

    displayBatchesEvaluated(totalBatchesEvaluated) {
        this.examplesEvaluated = batchSize * totalBatchesEvaluated;
        this.batchesEvaluatedElt.innerHTML = `Examples evaluated: ${this.examplesEvaluated}`;
        this.evalProgressBar.style.width = (this.examplesEvaluated / this.targetEvalExampleAmount) * 100 + '%';

        if (this.examplesEvaluated > 0) {
            this.evalProgressBarBackground.style.visibility = 'visible';
        }
        if (this.examplesEvaluated >= this.targetEvalExampleAmount) {
            this.graphRunner.totalBatchesEvaluated = 0;
            this.btn_eval.click();
        }

    }

    displayEvalExamplesPerSec(_examplesPerSec) {

        this.evalExamplesPerSec =
            smoothExamplesPerSec(this.evalExamplesPerSec, _examplesPerSec);

        this.examplesPerSecElt.innerHTML = `Examples/sec: ${this.evalExamplesPerSec}`;
    }

    displayInferenceExamplesOutput(
        inputFeeds, inferenceOutputs) {

        // let realImages = [];
        let fakeImages = [];

        for (let i = 0; i < inferenceOutputs.length; i++) {
            // realImages.push(inputFeeds[i][0].data);
            fakeImages.push((inferenceOutputs[i]));

        }

        // realImages =
        //     dataSet.unnormalizeExamples(realImages, IMAGE_DATA_INDEX);

        fakeImages =
            dataSet.unnormalizeExamples(fakeImages, IMAGE_DATA_INDEX);

        for (let i = 0; i < inferenceOutputs.length; i++) {
            // ndarrayVisualizers[i].saveImageDataFromNDArray(realImages[i]);
            this.ndarrayVisualizers[i].saveImageDataFromNDArray(fakeImages[i]);
        }

        for (let i = 0; i < inferenceOutputs.length; i++) {
            // ndarrayVisualizers[i].draw();
            this.ndarrayVisualizers[i].draw();

        }
    }

    monitorEvalRequestAndUpdateUI() {

        if (this.eval_paused) {
            if (this.getBatchesEvaluated() > 0) {
                this.btn_eval.value = 'Resume';
            } else {
                this.btn_eval.value = 'Evaluate';
            }
        } else {
            this.btn_eval.value = 'Pause'
        }

        if (this.eval_request) {
            this.eval_request = false;

            if (this.getBatchesEvaluated() > 0) {
                this.resumeEvaluating();
            } else {

                if (this.chartData.length > 0) {
                    this.chartData = [];
                }
                this.startEvalulating();
            }

        }

    }

    createModel(loadedWeights = null) {

        this.modelInitialized = false;
        if (this.isValid === false) {
            console.log('returning');
            return;
        }

        // Construct graph
        this.graph = new Graph();
        const g = this.graph;
        if (this.needGen) {
            this.randomTensor = g.placeholder('random', this.generatorNet.inputShape);
        } else {
            this.x0Tensor = g.placeholder('inputSampleImage', this.criticNet.inputShape);
        }
        this.xTensor = g.placeholder('input', this.criticNet.inputShape);
        this.oneTensor = g.placeholder('one', [2]);
        this.zeroTensor = g.placeholder('zero', [2]);

        const varianceInitializer = new VarianceScalingInitializer()
        const zerosInitializer = new ZerosInitializer()
        const onesInitializer = new OnesInitializer();


        let weights = null;
        let gen = null;
        if (this.needGen) {
            if (loadedWeights != null) {

                function toArray(dicValues, dicSize) {
                    var array = dicValues;
                    array.length = dicSize;
                    return Array.prototype.slice.call(array);
                }

                console.log('loading weights', loadedWeights);
                weights = {};
                for (var key in loadedWeights) {
                    weights[key] = toArray(loadedWeights[key].ndarrayData.values, loadedWeights[key].size);
                }

            } else {
                console.log('no weights loaded, random initializing weights');
            }


            // Construct generator
            gen = this.randomTensor;
            for (let i = 0; i < this.generatorNet.hiddenLayers.length; i++) {
                [gen] = this.generatorNet.hiddenLayers[i].addLayerMultiple(g, [gen],
                    'generator-' + i.toString(), weights); // weights is a dictionary {w: NDArray, b: NDArray}
            }
            gen = g.tanh(gen);
        }


        // Construct critic
        let crit1 = null;
        if (this.needGen) {
            crit1 = gen;
            this.generatedImage = gen;
        } else {
            crit1 = this.x0Tensor; // the image tensor make sure not the same as xTensor
            this.generatedImage = this.x0Tensor;
        }

        let crit2 = this.xTensor; // real image
        for (let i = 0; i < this.criticNet.hiddenLayers.length; i++) {
            let _weights = null;
            // if (loadedWeights != null) {
            //     weights = loadedWeights[i];
            // } // always need to retrain critic (which is the process of eval), never load weights for critic
            [crit1, crit2] = this.criticNet.hiddenLayers[i].addLayerMultiple(g, [crit1, crit2],
                'critic-' + i.toString(), _weights);
        }

        this.critPredictionReal = crit2;
        this.critPredictionFake = crit1;

        let critLossReal = null;
        let critLossFake = null;
        if (this.metricName == 'js') {
            critLossReal = g.softmaxCrossEntropyCost(
                this.critPredictionReal,
                this.oneTensor
            );
            critLossFake = g.softmaxCrossEntropyCost(
                this.critPredictionFake,
                this.zeroTensor
            );
        } else if (this.metricName == 'ls') {
            critLossReal = g.meanSquaredCost(
                this.critPredictionReal,
                this.oneTensor
            );
            critLossFake = g.meanSquaredCost(
                this.critPredictionFake,
                this.zeroTensor
            );
        } else {
            critLossReal = g.softmaxCrossEntropyCost(
                this.critPredictionReal,
                this.oneTensor
            );
            critLossFake = g.softmaxCrossEntropyCost(
                this.critPredictionFake,
                this.zeroTensor
            );
            this.metricName = 'js'
        }
        this.critLoss = g.add(critLossReal, critLossFake); // js/ls divergence

        if (this.session != null) {
            this.session.dispose()
        }
        this.session = new Session(g, math);
        this.graphRunner.setSession(this.session);

        // startInference();

        this.modelInitialized = true;

        console.log('modelid', this.id, 'initialized = true');
    }

    getBatchesEvaluated() {
        return this.graphRunner.getTotalBatchesEvaluated()
    }

    startInference() {
        const data = getImageDataOnly(dataSet);
        if (data == null) {
            return;
        }
        if (this.isValid && (data != null)) {
            const shuffledInputProviderGenerator =
                new InCPUMemoryShuffledInputProviderBuilder([data]);
            const [inputImageProvider] =
            shuffledInputProviderGenerator.getInputProviders();

            let inputSampleImageProvider;
            if (!this.needGen) {
                if (this.sampleImage && this.sampleImageDataLoaded) {
                    const sampleImageData = getImageDataOnly(this.sampleImageDataSet);
                    const shuffledInputSampleImageProviderGenerator =
                        new InCPUMemoryShuffledInputProviderBuilder([sampleImageData]);
                    [inputSampleImageProvider] =
                    shuffledInputSampleImageProviderGenerator.getInputProviders();
                } else {
                    [inputSampleImageProvider] =
                    shuffledInputProviderGenerator.getInputProviders();
                }
            }


            const oneInputProvider = {
                getNextCopy(math) {
                    return Array1D.new([0, 1]);
                },
                disposeCopy(math, copy) {
                    copy.dispose();
                }
            }

            const zeroInputProvider = {
                getNextCopy(math) {
                    return Array1D.new([1, 0]);
                },
                disposeCopy(math, copy) {
                    copy.dispose();
                }
            }

            const inferenceFeeds = [{
                    tensor: this.xTensor,
                    data: inputImageProvider
                },
                {
                    tensor: this.needGen ? this.randomTensor : this.x0Tensor,
                    data: this.needGen ? getRandomInputProvider(this.generatorNet.inputShape) : inputSampleImageProvider
                },
                {
                    tensor: this.oneTensor,
                    data: oneInputProvider
                },
                {
                    tensor: this.zeroTensor,
                    data: zeroInputProvider
                }
            ]

            this.graphRunner.infer(
                this.generatedImage, null, null,
                inferenceFeeds, INFERENCE_EXAMPLE_INTERVAL_MS, INFERENCE_EXAMPLE_COUNT
            );
        }
    }

    stopInference() {

        this.graphRunner.stopInferring();
    }

    stopEvaluating() {

        this.graphRunner.stopEvaluating();
    }

    resumeEvaluating() {
        this.graphRunner.resumeEvaluating();
    }

    startEvalulating() {
        const data = getImageDataOnly(dataSet);

        // Recreate optimizer with the selected optimizer and hyperparameters.
        let critOptimizer = createOptimizer('crit', this.graph); // for js, exact same optimizer
        // genOptimizer = createOptimizer('gen');

        if (this.isValid && data != null) {
            // recreateCharts();
            this.graphRunner.resetStatistics();

            const shuffledInputProviderGenerator =
                new InCPUMemoryShuffledInputProviderBuilder([data]);
            const [inputImageProvider] =
            shuffledInputProviderGenerator.getInputProviders();

            let inputSampleImageProvider;
            if (!this.needGen) {
                if (this.sampleImage && this.sampleImageDataLoaded) {
                    const sampleImageData = getImageDataOnly(this.sampleImageDataSet);
                    const shuffledInputSampleImageProviderGenerator =
                        new InCPUMemoryShuffledInputProviderBuilder([sampleImageData]);
                    [inputSampleImageProvider] =
                    shuffledInputSampleImageProviderGenerator.getInputProviders();
                } else {
                    [inputSampleImageProvider] =
                    shuffledInputProviderGenerator.getInputProviders();
                }
            }

            const oneInputProvider = {
                getNextCopy(math) {
                    return Array1D.new([0, 1]);
                },
                disposeCopy(math, copy) {
                    copy.dispose();
                }
            }

            const zeroInputProvider = {
                getNextCopy(math) {
                    return Array1D.new([1, 0]);
                },
                disposeCopy(math, copy) {
                    copy.dispose();
                }
            }

            const critFeeds = [{
                    tensor: this.xTensor,
                    data: inputImageProvider
                },
                {
                    tensor: this.needGen ? this.randomTensor : this.x0Tensor,
                    data: this.needGen ? getRandomInputProvider(this.generatorNet.inputShape) : inputSampleImageProvider
                },
                {
                    tensor: this.oneTensor,
                    data: oneInputProvider
                },
                {
                    tensor: this.zeroTensor,
                    data: zeroInputProvider
                }
            ]

            this.graphRunner.evaluate(
                this.critLoss, null, critFeeds, null, batchSize,
                critOptimizer, null, undefined, COST_INTERVAL_MS);

            // showEvalStats = true;
            // applicationState = ApplicationState.Evaluating;
        }
    }

}

function getImageDataOnly(dataSet) {
    const [images, labels] = dataSet.getData();
    return images
}

function getRandomInputProvider(shape) {
    return {
        getNextCopy(math) {
            return NDArray.randNormal(shape);
        },
        disposeCopy(math, copy) {
            copy.dispose();
        }
    }
}

function createOptimizer(which, graph) {
    if (which === 'gen') {
        // var selectedOptimizerName = genSelectedOptimizerName;
        // var learningRate = genLearningRate;
        // var momentum = genMomentum;
        // var gamma = genGamma;
        // var beta1 = genBeta1;
        // var beta2 = genBeta2;
        // var varName = 'generator';
    } else { // critic
        var selectedOptimizerName = critSelectedOptimizerName;
        var learningRate = critLearningRate;
        var momentum = critMomentum;
        var gamma = critGamma;
        var beta1 = critBeta1;
        var beta2 = critBeta2;
        var varName = 'critic';
    }
    switch (selectedOptimizerName) {
        case 'sgd':
            {
                return new SGDOptimizer(+learningRate,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'momentum':
            {
                return new MomentumOptimizer(+learningRate, +momentum,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'rmsprop':
            {
                return new RMSPropOptimizer(+learningRate, +gamma,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adagrad':
            {
                return new AdagradOptimizer(+learningRate,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adadelta':
            {
                return new AdadeltaOptimizer(+learningRate, +gamma,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adam':
            {
                return new AdamOptimizer(+learningRate, +beta1, +beta2,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        default:
            {
                throw new Error(`Unknown optimizer`);
            }
    }
}