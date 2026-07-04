// charts.js - ApexCharts initialization and management module

let chartEar = null;
let chartMar = null;
let chartPose = null;

const commonOptions = {
    chart: {
        type: 'line',
        height: '100%',
        animations: {
            enabled: false
        },
        toolbar: { show: false },
        sparkline: { enabled: false },
        background: 'transparent'
    },
    stroke: { curve: 'smooth', width: 2 },
    grid: {
        borderColor: 'rgba(51, 65, 85, 0.15)',
        strokeDashArray: 2,
        xaxis: { lines: { show: true } },
        yaxis: { lines: { show: true } }
    },
    theme: { mode: 'dark' },
    tooltip: {
        enabled: true,
        theme: 'dark',
        x: { show: false },
        marker: { show: false },
        y: {
            formatter: function(val) {
                return val !== null && val !== undefined ? val.toFixed(3) : '';
            }
        },
        style: {
            fontSize: '10px',
            fontFamily: 'JetBrains Mono'
        }
    },
    legend: { show: false }
};

export function initCharts() {
    if (typeof ApexCharts === 'undefined') {
        console.warn("ApexCharts is not defined. Dashboard charts will be disabled.");
        return;
    }
    // EAR Chart
    const optionsEar = {
        ...commonOptions,
        series: [{ name: 'EAR', data: [] }],
        colors: ['#06b6d4'],
        yaxis: { 
            min: 0, 
            max: 0.5, 
            tickAmount: 5, 
            labels: { 
                formatter: (val) => (val !== undefined && val !== null && !isNaN(val)) ? Number(val).toFixed(2) : "",
                style: { colors: '#94a3b8', fontSize: '9px' } 
            },
            title: {
                text: 'Eye Aspect Ratio (EAR)',
                style: { color: '#94a3b8', fontSize: '9px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
            }
        },
        xaxis: { 
            labels: { show: false },
            title: {
                text: 'Time Timeline (seconds)',
                style: { color: '#64748b', fontSize: '9px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
            }
        },
        annotations: {
            yaxis: [{
                y: 0.22,
                borderColor: '#ef4444',
                strokeDashArray: 3,
                label: {
                    borderColor: '#ef4444',
                    style: { color: '#fff', background: '#ef4444', fontSize: '9px' },
                    text: 'Threshold (0.22)'
                }
            }]
        }
    };
    chartEar = new ApexCharts(document.querySelector("#chart-ear"), optionsEar);
    chartEar.render();

    // MAR Chart
    const optionsMar = {
        ...commonOptions,
        series: [{ name: 'MAR', data: [] }],
        colors: ['#a855f7'],
        yaxis: { 
            min: 0, 
            max: 1.0, 
            tickAmount: 4, 
            labels: { 
                formatter: (val) => (val !== undefined && val !== null && !isNaN(val)) ? Number(val).toFixed(2) : "",
                style: { colors: '#94a3b8', fontSize: '9px' } 
            },
            title: {
                text: 'Mouth Aspect Ratio (MAR)',
                style: { color: '#94a3b8', fontSize: '9px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
            }
        },
        xaxis: { 
            labels: { show: false },
            title: {
                text: 'Time Timeline (seconds)',
                style: { color: '#64748b', fontSize: '9px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
            }
        },
        annotations: {
            yaxis: [{
                y: 0.60,
                borderColor: '#ef4444',
                strokeDashArray: 3,
                label: {
                    borderColor: '#ef4444',
                    style: { color: '#fff', background: '#ef4444', fontSize: '9px' },
                    text: 'Yawn Threshold (0.60)'
                }
            }]
        }
    };
    chartMar = new ApexCharts(document.querySelector("#chart-mar"), optionsMar);
    chartMar.render();

    // Head Pose Chart
    const optionsPose = {
        ...commonOptions,
        series: [
            { name: 'Pitch', data: [] },
            { name: 'Yaw', data: [] },
            { name: 'Roll', data: [] }
        ],
        colors: ['#f59e0b', '#3b82f6', '#10b981'],
        yaxis: { 
            min: -50, 
            max: 50, 
            tickAmount: 4, 
            labels: { 
                formatter: (val) => (val !== undefined && val !== null && !isNaN(val)) ? Number(val).toFixed(0) : "",
                style: { colors: '#94a3b8', fontSize: '9px' } 
            },
            title: {
                text: 'Orientation Angle (degrees)',
                style: { color: '#94a3b8', fontSize: '9px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
            }
        },
        xaxis: { 
            labels: { show: false },
            title: {
                text: 'Time Timeline (seconds)',
                style: { color: '#64748b', fontSize: '9px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
            }
        },
        legend: {
            show: true,
            position: 'top',
            horizontalAlign: 'right',
            fontSize: '9px',
            labels: { colors: '#94a3b8' },
            itemMargin: { horizontal: 5, vertical: 0 }
        }
    };
    chartPose = new ApexCharts(document.querySelector("#chart-pose"), optionsPose);
    chartPose.render();
}

export function updateCharts(earHistory, marHistory, poseHistory) {
    if (typeof ApexCharts === 'undefined') return;
    if (chartEar) chartEar.updateSeries([{ name: 'EAR', data: earHistory }]);
    if (chartMar) chartMar.updateSeries([{ name: 'MAR', data: marHistory }]);
    if (chartPose) {
        chartPose.updateSeries([
            { name: 'Pitch', data: poseHistory.pitch },
            { name: 'Yaw', data: poseHistory.yaw },
            { name: 'Roll', data: poseHistory.roll }
        ]);
    }
}


let chartEnlarged = null;
let enlargedChartType = null;

export function resetCharts() {
    if (typeof ApexCharts === 'undefined') return;
    if (chartEar) chartEar.updateSeries([{ name: 'EAR', data: [] }]);
    if (chartMar) chartMar.updateSeries([{ name: 'MAR', data: [] }]);
    if (chartPose) {
        chartPose.updateSeries([
            { name: 'Pitch', data: [] },
            { name: 'Yaw', data: [] },
            { name: 'Roll', data: [] }
        ]);
    }
    if (chartEnlarged && enlargedChartType) {
        if (enlargedChartType === 'ear') chartEnlarged.updateSeries([{ name: 'EAR', data: [] }]);
        else if (enlargedChartType === 'mar') chartEnlarged.updateSeries([{ name: 'MAR', data: [] }]);
        else if (enlargedChartType === 'pose') {
            chartEnlarged.updateSeries([
                { name: 'Pitch', data: [] },
                { name: 'Yaw', data: [] },
                { name: 'Roll', data: [] }
            ]);
        }
    }
}

export function showEnlargedChart(type, earHistory, marHistory, poseHistory) {
    if (typeof ApexCharts === 'undefined') return;
    
    // Destroy previous enlarged chart if any
    if (chartEnlarged) {
        chartEnlarged.destroy();
        chartEnlarged = null;
    }
    
    enlargedChartType = type;
    
    // Configure options based on selected type
    let options = null;
    if (type === 'ear') {
        options = {
            ...commonOptions,
            chart: {
                ...commonOptions.chart,
                height: '100%'
            },
            series: [{ name: 'EAR', data: earHistory }],
            colors: ['#06b6d4'],
            yaxis: { 
                min: 0, 
                max: 0.5, 
                tickAmount: 5, 
                labels: { 
                    formatter: (val) => (val !== undefined && val !== null && !isNaN(val)) ? Number(val).toFixed(2) : "",
                    style: { colors: '#94a3b8', fontSize: '11px' } 
                },
                title: {
                    text: 'Eye Aspect Ratio (EAR)',
                    style: { color: '#94a3b8', fontSize: '11px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
                }
            },
            xaxis: { 
                labels: { show: false },
                title: {
                    text: 'Time Timeline (seconds)',
                    style: { color: '#64748b', fontSize: '11px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
                }
            },
            annotations: {
                yaxis: [{
                    y: 0.22,
                    borderColor: '#ef4444',
                    strokeDashArray: 3,
                    label: {
                        borderColor: '#ef4444',
                        style: { color: '#fff', background: '#ef4444', fontSize: '10px' },
                        text: 'Threshold (0.22)'
                    }
                }]
            }
        };
    } else if (type === 'mar') {
        options = {
            ...commonOptions,
            chart: {
                ...commonOptions.chart,
                height: '100%'
            },
            series: [{ name: 'MAR', data: marHistory }],
            colors: ['#a855f7'],
            yaxis: { 
                min: 0, 
                max: 1.0, 
                tickAmount: 4, 
                labels: { 
                    formatter: (val) => (val !== undefined && val !== null && !isNaN(val)) ? Number(val).toFixed(2) : "",
                    style: { colors: '#94a3b8', fontSize: '11px' } 
                },
                title: {
                    text: 'Mouth Aspect Ratio (MAR)',
                    style: { color: '#94a3b8', fontSize: '11px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
                }
            },
            xaxis: { 
                labels: { show: false },
                title: {
                    text: 'Time Timeline (seconds)',
                    style: { color: '#64748b', fontSize: '11px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
                }
            },
            annotations: {
                yaxis: [{
                    y: 0.60,
                    borderColor: '#ef4444',
                    strokeDashArray: 3,
                    label: {
                        borderColor: '#ef4444',
                        style: { color: '#fff', background: '#ef4444', fontSize: '10px' },
                        text: 'Yawn Threshold (0.60)'
                    }
                }]
            }
        };
    } else if (type === 'pose') {
        options = {
            ...commonOptions,
            chart: {
                ...commonOptions.chart,
                height: '100%'
            },
            series: [
                { name: 'Pitch', data: poseHistory.pitch },
                { name: 'Yaw', data: poseHistory.yaw },
                { name: 'Roll', data: poseHistory.roll }
            ],
            colors: ['#f59e0b', '#3b82f6', '#10b981'],
            yaxis: { 
                min: -50, 
                max: 50, 
                tickAmount: 4, 
                labels: { 
                    formatter: (val) => (val !== undefined && val !== null && !isNaN(val)) ? Number(val).toFixed(0) : "",
                    style: { colors: '#94a3b8', fontSize: '11px' } 
                },
                title: {
                    text: 'Orientation Angle (degrees)',
                    style: { color: '#94a3b8', fontSize: '11px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
                }
            },
            xaxis: { 
                labels: { show: false },
                title: {
                    text: 'Time Timeline (seconds)',
                    style: { color: '#64748b', fontSize: '11px', fontFamily: 'JetBrains Mono', fontWeight: 600 }
                }
            },
            legend: {
                show: true,
                position: 'top',
                horizontalAlign: 'right',
                fontSize: '11px',
                labels: { colors: '#94a3b8' },
                itemMargin: { horizontal: 5, vertical: 0 }
            }
        };
    }
    
    if (options) {
        chartEnlarged = new ApexCharts(document.querySelector("#chart-enlarged"), options);
        chartEnlarged.render();
    }
}

export function closeEnlargedChart() {
    if (chartEnlarged) {
        chartEnlarged.destroy();
        chartEnlarged = null;
    }
    enlargedChartType = null;
}

export function updateEnlargedChart(earHistory, marHistory, poseHistory) {
    if (!chartEnlarged || !enlargedChartType) return;
    
    if (enlargedChartType === 'ear') {
        chartEnlarged.updateSeries([{ name: 'EAR', data: earHistory }]);
    } else if (enlargedChartType === 'mar') {
        chartEnlarged.updateSeries([{ name: 'MAR', data: marHistory }]);
    } else if (enlargedChartType === 'pose') {
        chartEnlarged.updateSeries([
            { name: 'Pitch', data: poseHistory.pitch },
            { name: 'Yaw', data: poseHistory.yaw },
            { name: 'Roll', data: poseHistory.roll }
        ]);
    }
}
