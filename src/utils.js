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

class NDArrayImageVisualizer {

    constructor(elt) {
        this.elt = elt;

        this.imageData = null;

        this.canvas = document.createElement('canvas'); // this.querySelector('#canvas');
        this.canvas.style.display = "table-cell";
        this.canvas.width = 0;
        this.canvas.height = 0;
        this.canvasContext =
            this.canvas.getContext('2d');
        this.canvas.style.display = 'none';
        this.elt.appendChild(this.canvas);
    }

    setShape(shape) {
        this.canvas.width = shape[1];
        this.canvas.height = shape[0];
    }

    setSize(width, height) {
        this.canvas.style.width = `${width}px`;
        this.canvas.style.height = `${height}px`;
    }

    saveImageDataFromNDArray(ndarray) {
        this.imageData = this.canvasContext.createImageData(
            this.canvas.width, this.canvas.height);
        if (ndarray.shape[2] === 1) {
            this.drawGrayscaleImageData(ndarray);
        } else if (ndarray.shape[2] === 3) {
            this.drawRGBImageData(ndarray);
        }
    }

    drawRGBImageData(ndarray) {
        let pixelOffset = 0;
        for (let i = 0; i < ndarray.shape[0]; i++) {
            for (let j = 0; j < ndarray.shape[1]; j++) {
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 0);
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 1);
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 2);
                this.imageData.data[pixelOffset++] = 255;
            }
        }
    }

    drawGrayscaleImageData(ndarray) {
        let pixelOffset = 0;
        for (let i = 0; i < ndarray.shape[0]; i++) {
            for (let j = 0; j < ndarray.shape[1]; j++) {
                const value = ndarray.get(i, j, 0);
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = 255;
            }
        }
    }

    draw() {
        this.canvas.style.display = '';
        this.canvasContext.putImageData(this.imageData, 0, 0);
    }
}

function indexOfDropdownOptions(options, selectedName) {

    for (var i = 0; i < options.length; i++) {
        if (options[i].value === selectedName) {
            return i
        }

    }
    console.assert(false, 'can not find selected option in option array');
}


// var chartData = [];

window.chartColors = {
    red: 'rgb(255, 99, 132)',
    orange: 'rgb(255, 159, 64)',
    yellow: 'rgb(255, 205, 86)',
    green: 'rgb(75, 192, 192)',
    blue: 'rgb(54, 162, 235)',
    purple: 'rgb(153, 102, 255)',
    grey: 'rgb(227, 227, 227)',
    greenblue: 'rgba(75,192,192,1)',
    transparent: 'rgba(255,255,255,1)'
};

var chartDataSets = [{
    label: "Test",
    backgroundColor: window.chartColors.greenblue,
    borderColor: window.chartColors.greenblue,
    data: [],
    fill: false,
    pointRadius: 0,
    // pointHitRadius: 5,
    borderWidth: 2,
    lineTension: 0,
}, {
    label: 'WorstPt',
    backgroundColor: window.chartColors.transparent,
    borderColor: window.chartColors.red,
    borderWidth: 3,
    data: [],
    fill: false,
    pointRadius: 5,
    pointHitRadius: 5
    // lineTension: 0,
}, {
    label: 'Train',
    backgroundColor: window.chartColors.grey,
    borderColor: window.chartColors.grey,
    data: [],
    fill: false,
    pointRadius: 0,
    // pointHitRadius: 5,
    borderWidth: 1,
    // lineTension: 0,
}]

var config = {
    type: 'line',
    data: {
        datasets: chartDataSets
    },
    options: {
        animation: {
            duration: 0
        },
        responsive: true,
        scales: {
            xAxes: [{
                display: true,
                type: 'linear',
                position: 'bottom',
                scalelabel: {
                    display: true,
                    labelString: 'iterations'
                }
            }],
            yAxes: [{
                display: true,
                ticks: {
                    min: null,
                    callback: (label, index, labels) => {
                        let num = Number(label).toFixed(2);
                        return `${num}`;
                    }
                },
                scalelabel: {
                    display: true,
                    labelString: 'cost'
                }
            }]
        }
    }
};





function createChart(canvasElt, label) {

    const context = canvasElt.getContext('2d');

    // config.data.datasets[2].label = label + `-` + config.data.datasets[2].label;
    // config.data.datasets[0].label = label + `-` + config.data.datasets[0].label;

    return new Chart(context, config);

}