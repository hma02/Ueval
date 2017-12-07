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
var Array1D = dl.Array1D;
var Array3D = dl.Array3D;
var DataStats = dl.DataStats;
var FeedEntry = dl.FeedEntry;
var Graph = dl.Graph;
var InCPUMemoryShuffledInputProviderBuilder = dl.InCPUMemoryShuffledInputProviderBuilder;
var Initializer = dl.Initializer;
var InMemoryDataset = dl.InMemoryDataset;
var MetricReduction = dl.MetricReduction;
// var MomentumOptimizer = dl.MomentumOptimizer;
// var SGDOptimizer = dl.SGDOptimizer;
// var RMSPropOptimizer = dl.RMSPropOptimizer;
// var AdagradOptimizer = dl.AdagradOptimizer;
// var AdadeltaOptimizer = dl.AdadeltaOptimizer;
var AdamOptimizer = dl.AdamOptimizer;
// var AdamaxOptimizer = dl.AdamaxOptimizer;
var NDArray = dl.NDArray;
var NDArrayMath = dl.NDArrayMath;
var NDArrayMathCPU = dl.NDArrayMathCPU;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Optimizer = dl.Optimizer;
var OnesInitializer = dl.OnesInitializer;
var Scalar = dl.Scalar;
var Session = dl.Session;
var Tensor = dl.Tensor;
var util = dl.util;
var VarianceScalingInitializer = dl.VarianceScalingInitializer;
var xhr_dataset = dl.xhr_dataset;
var XhrDataset = dl.XhrDataset;
var XhrDatasetConfig = dl.XhrDatasetConfig;
var ZerosInitializer = dl.ZerosInitializer;

const INITIALIZATION_RANDOM_SEED = 1234;

const DATASETS_CONFIG_JSON = 'src/model-builder-datasets-config.json';

/** How often to evaluate the model against test data. */
const EVAL_INTERVAL_MS = 1500;
/** How often to compute the cost. Downloading the cost stalls the GPU. */
const COST_INTERVAL_MS = 500;
/** How many inference examples to show when evaluating accuracy. */
const INFERENCE_EXAMPLE_COUNT = 9; // must be a square number
const INFERENCE_EXAMPLE_ROWS = Math.sqrt(INFERENCE_EXAMPLE_COUNT);
const INFERENCE_IMAGE_SIZE_PX = 50;
/**
 * How often to show inference examples. This should be less often than
 * EVAL_INTERVAL_MS as we only show inference examples during an eval.
 */
const INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

const TARGET_EVALUATION_EXAMPLE_AMOUNT = 40000;

// Smoothing factor for the examples/s standalone text statistic.
const EXAMPLE_SEC_STAT_SMOOTHING_FACTOR = .7;

const TRAIN_TEST_RATIO = 5 / 6;

const IMAGE_DATA_INDEX = 0;
const LABEL_DATA_INDEX = 1;

const metricNames = {
    js: 'js', // Jensen-Shannon divergence
    ls: 'ls' // least square divergence
}

var METRIC_NAME = metricNames.js;

var datasetDownloaded;
var dataSet;

var critLearningRate = 0.01;
var critMomentum = 0.1;
var critNeedMomentum = false;
var critGamma = 0.1;
var critBeta1 = 0.9;
var critBeta2 = 0.999;
var critNeedGamma = false;
var critNeedBeta = false;
var batchSize = 30;
var critSelectedOptimizerName;

var selectedEnvName;
var math;
// Keep one instance of each NDArrayMath so we don't create a user-initiated
// number of NDArrayMathGPU's.
var mathGPU = new NDArrayMathGPU();
var mathCPU = new NDArrayMathCPU();

var applicationState;

// this is a global function for preparing the datasets for all models within this application
var selectedDatasetName;
var xhrDatasetConfigs;

function fetchConfig_DownloadData(fetchConfigCallback) {
    var dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON).then(
        _xhrDatasetConfigs => {

            for (const datasetName in _xhrDatasetConfigs) {
                if (_xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                    dataSets[datasetName] =
                        new XhrDataset(_xhrDatasetConfigs[datasetName]);
                }
            }
            var datasetNames = Object.keys(dataSets);
            selectedDatasetName = datasetNames[0]; // 0: MNIST 1: FashionMNIST 2:Runes 3: CIFAR10

            dataSet = dataSets[selectedDatasetName];
            xhrDatasetConfigs = _xhrDatasetConfigs;

            fetchConfigCallback(xhrDatasetConfigs, selectedDatasetName);

            datasetDownloaded = false;

            dataSet.fetchData().then(() => {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
                datasetDownloaded = true;
            });

        },
        error => {
            throw new Error('Dataset config could not be loaded: ' + error);
        });
}

// -------- global function to build all needed models within the application

var models = [];

function buildModels(xhrDatasetConfigs, selectedDatasetName) {
    const configs = xhrDatasetConfigs[selectedDatasetName];

    var evalModelRealImage = new EvalSampleModel(configs, models.length, false, true);
    evalModelRealImage.initialize();
    models.push(evalModelRealImage);

    var evalModel = new EvalSampleModel(configs, models.length);
    evalModel.initialize();
    models.push(evalModel);

}

// --------------------  display and control  -------------------------------

function updateSelectedEnvironment(selectedEnvName, _graphRunner = null) {
    math = (selectedEnvName === 'GPU') ? mathGPU : mathCPU;
    if (_graphRunner != null) {
        console.log('math =', math === mathGPU ? 'mathGPU' : 'mathCPU', 'with graphRunner', _graphRunner);
        _graphRunner.setMath(math);
    } else {
        if (models.length > 0) {
            models.forEach(m => m.graphRunner.setMath(math))
            console.log('math =', math === mathGPU ? 'mathGPU' : 'mathCPU', 'with graphRunners in all models', );
        } else {
            console.log('math =', math === mathGPU ? 'mathGPU' : 'mathCPU');
        }
    }
}

var infer_request = null;
var btn_infer = document.getElementById('buttoninfer');
var infer_paused = true;
btn_infer.addEventListener('click', () => {

    infer_paused = !infer_paused;
    if (infer_paused) {
        if (models.length > 0) {
            models.forEach(m => m.stopInference()); // can return quickly
        }
    } else {
        infer_request = true;
    }

    ga('send', 'event', 'Ueval', 'click', 'Infer All', 40);

});

var ulBtns = document.querySelectorAll(".upload");

var targetEvalAmountElt = document.querySelector("#targetEvalAmount");
targetEvalAmountElt.innerHTML = `Target Eval Amount: ${TARGET_EVALUATION_EXAMPLE_AMOUNT}`;


// ----------------------- application initialization and monitor ----------------------


function run() {

    critLearningRate = 0.01;
    critMomentum = 0.1;
    critNeedMomentum = false;
    critGamma = 0.1;
    critBeta1 = 0.9;
    critBeta2 = 0.999;
    critNeedGamma = false;
    critNeedBeta = false;
    batchSize = 30;
    // Default optimizer is momentum
    critSelectedOptimizerName = "adam";

    var envDropdown = document.getElementById("environment-dropdown");
    selectedEnvName = 'GPU';
    var ind = indexOfDropdownOptions(envDropdown.options, selectedEnvName)
    envDropdown.options[ind].selected = 'selected';
    updateSelectedEnvironment(selectedEnvName); // change math

    envDropdown.addEventListener('change', (event) => {
        selectedEnvName = event.target.value;
        updateSelectedEnvironment(selectedEnvName); // change math
    });


    var metricDropdown = document.getElementById("metric-dropdown");
    METRIC_NAME = metricNames.js;
    var ind = indexOfDropdownOptions(metricDropdown.options, METRIC_NAME)
    metricDropdown.options[ind].selected = 'selected';

    metricDropdown.addEventListener('change', (event) => {
        METRIC_NAME = event.target.value;
        models = [];
        if (xhrDatasetConfigs != null && selectedDatasetName != null) {
            buildModels(xhrDatasetConfigs, selectedDatasetName)
        }
    });


    // Set up datasets.
    fetchConfig_DownloadData(buildModels);

}

function monitor() {

    if (datasetDownloaded == false) {
        btn_infer.disabled = true;
        btn_infer.className = 'btn btn-default btn-md paper';
        btn_infer.value = 'Downloading data ...';
        models.forEach(m => m.btn_eval.disabled = true);
        ulBtns.forEach(elt => elt.disabled = true);

    } else {
        if (models.every(m => m.isValid)) {
            if (models.every(m => m.modelInitialized)) {

                btn_infer.className = 'btn btn-default btn-md paper';
                btn_infer.disabled = false;

                models.forEach(m => m.btn_eval.disabled = false);
                ulBtns.forEach(elt => elt.disabled = false);

                if (infer_paused) {
                    btn_infer.value = 'Infer All'
                } else {
                    btn_infer.value = 'Stop Infering'
                }

                if (infer_request) {
                    infer_request = false;
                    models.forEach(m => m.startInference());
                }

                models.forEach(m => m.monitorEvalRequestAndUpdateUI());

            } else {
                btn_infer.className = 'btn btn-default btn-md paper';
                btn_infer.disabled = true;
                btn_infer.value = 'Initializing Model ...'

            }
        } else {
            btn_infer.className = 'btn btn-danger btn-md paper';
            btn_infer.disabled = true;
            btn_infer.value = 'Model not valid'
        }
    }

    setTimeout(function () {
        monitor();
    }, 100);
}

function start() {

    supported = detect_support();

    var inputs = document.getElementsByTagName("input");

    if (supported) {
        console.log('device & webgl supported');

        for (var i = 0; i < inputs.length; i++) {
            if (inputs[i].type === 'submit') {
                inputs[i].disabled = false;
            }
        }

        setTimeout(function () {
            run(); // initialize data and model
            monitor(); // monitor button clicks and display update
        }, 0);

    } else {
        console.log('device/webgl not supported')

        for (var i = 0; i < inputs.length; i++) {
            if (inputs[i].type === 'submit') {
                inputs[i].disabled = true;
            }
        }
    }
}