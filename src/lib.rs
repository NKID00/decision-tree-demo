mod tree;

use tree::Class;
use tree::*;

use std::{
    collections::{HashMap, HashSet},
    iter::repeat_n,
};

use bimap::BiMap;
use js_sys::{Array, Object, Reflect};
use leptos::logging::log;
use leptos::*;
use leptos_dom::helpers::{get_property, set_property};
use leptos_meta::*;
use rand::{seq::IteratorRandom, SeedableRng};
use rand_chacha::ChaCha12Rng;
use stylers::style_str;
use wasm_bindgen::{prelude::*, JsValue};
use web_sys::Element;

const RANDOM_SEED: u64 = 0;

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();
    view! {
        <ErrorBoundary fallback=|errors| {
            view! {
                <h1>"Uh oh! Something went wrong!"</h1>
                <p>"Errors: "</p>
                <ul>
                    {move || {
                        errors
                            .get()
                            .into_iter()
                            .map(|(_, e)| view! { <li>{e.to_string()}</li> })
                            .collect_view()
                    }}

                </ul>
            }
        }>
            <Main />
        </ErrorBoundary>
    }
}

#[derive(Debug, Clone)]
struct DataSet {
    axises: Vec<String>,
    classes: Vec<String>,
    axis_map: BiMap<String, Axis>,
    class_map: BiMap<String, Class>,
    mapped_axis: Vec<Axis>,
    mapped_class: Vec<Class>,
    rows: Vec<(Vec<f64>, String)>,
}

fn load_csv(source: String) -> &'static str {
    match source.as_str() {
        "iris" => include_str!("../public/iris.csv"),
        "wine" => include_str!("../public/wine.csv"),
        "rice" => include_str!("../public/rice.csv"),
        _ => unreachable!(),
    }
}

fn load_dataset(csv: &'static str) -> DataSet {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv.as_bytes());
    let mut axises: Vec<String> = rdr.headers().unwrap().iter().map(str::to_owned).collect();
    axises.pop().unwrap();
    let mut classes = HashSet::<String>::new();
    let rows = rdr
        .records()
        .map(|record| {
            let record = record.unwrap();
            let mut vec: Vec<&str> = record.iter().collect();
            let last = vec.pop().unwrap();
            let last = last.to_owned();
            classes.insert(last.clone());
            (vec.iter().map(|v| v.parse().unwrap()).collect(), last)
        })
        .collect();
    let axis_map = BiMap::from_iter(axises.iter().map(|s| s.to_owned()).zip((0..).map(Axis)));
    let class_map = BiMap::from_iter(classes.iter().map(|s| s.to_owned()).zip((0..).map(Class)));
    let mapped_axis = axises
        .iter()
        .map(|axis| *axis_map.get_by_left(axis).unwrap())
        .collect();
    let mapped_class = classes
        .iter()
        .map(|class| *class_map.get_by_left(class).unwrap())
        .collect();
    DataSet {
        axises,
        classes: classes.into_iter().collect(),
        axis_map,
        class_map,
        mapped_axis,
        mapped_class,
        rows,
    }
}

fn get(obj: &JsValue, prop: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(prop)).unwrap()
}

fn set(obj: &JsValue, prop: &str, value: &JsValue) {
    Reflect::set(obj, &JsValue::from_str(prop), value).unwrap();
}

#[wasm_bindgen(
    inline_js = "export function new_chart(ctx, config) { return new Chart(ctx, config); }"
)]
extern "C" {
    fn new_chart(ctx: &web_sys::HtmlElement, config: &Object) -> JsValue;
}

#[wasm_bindgen(inline_js = "export function update_chart(chart) { chart.update(); }")]
extern "C" {
    fn update_chart(chart: &JsValue);
}

#[wasm_bindgen(inline_js = "export function toast_alert_ffi(alert) { alert.toast(); }")]
extern "C" {
    fn toast_alert_ffi(alert: &JsValue);
}

fn toast_alert(alert: NodeRef<html::Custom>) {
    toast_alert_ffi(&(alert.get_untracked().unwrap().into_any()));
}

fn split_train_classify_dataset(
    dataset: DataSet,
) -> (Vec<(DataPoint, Class)>, Vec<DataPoint>, Vec<DataPoint>) {
    let mut rng = ChaCha12Rng::seed_from_u64(RANDOM_SEED);
    let rows = dataset.rows;
    let len = rows.len();
    let mut classify_index = (0..len).choose_multiple(&mut rng, len / 5);
    classify_index.sort();
    (
        rows.iter()
            .enumerate()
            .filter_map(|(i, row)| {
                if classify_index.binary_search(&i).is_err() {
                    Some((
                        DataPoint(row.0.clone()),
                        *dataset.class_map.get_by_left(&row.1).unwrap(),
                    ))
                } else {
                    None
                }
            })
            .collect(),
        rows.iter()
            .enumerate()
            .filter_map(|(i, row)| {
                if classify_index.binary_search(&i).is_err() {
                    Some(DataPoint(row.0.clone()))
                } else {
                    None
                }
            })
            .collect(),
        rows.iter()
            .enumerate()
            .filter_map(|(i, row)| {
                if classify_index.binary_search(&i).is_ok() {
                    Some(DataPoint(row.0.clone()))
                } else {
                    None
                }
            })
            .collect(),
    )
}

fn into_chart_dataset(
    x_axis: usize,
    y_axis: usize,
    classes: &[String],
    rows: &[(Vec<f64>, String)],
) -> JsValue {
    let obj = Object::new();
    set(
        &obj,
        "datasets",
        &{
            let arr = Array::from_iter(classes.iter().map(|class| {
                let obj = Object::new();
                set(&obj, "label", &class.into());
                set(
                    &obj,
                    "data",
                    &Array::from_iter(rows.iter().filter_map(|(vec, label)| {
                        if label == class {
                            let obj = Object::new();
                            set(&obj, "x", &vec[x_axis].into());
                            set(&obj, "y", &vec[y_axis].into());
                            Some(obj)
                        } else {
                            None
                        }
                    })),
                );
                obj
            }));
            arr
        }
        .into(),
    );
    obj.into()
}

fn display_tree(dataset: &DataSet, tree: &DecisionTree, indent: usize) -> String {
    let indent_s = repeat_n(' ', indent * 2).collect::<String>();
    match tree {
        DecisionTree::Branch(c, left_tree, right_tree) => {
            let left = display_tree(dataset, left_tree, indent + 1);
            let right = display_tree(dataset, right_tree, indent + 1);
            format!(
                "{}if {} < {:.3} {{\n{}\n{}}} else {{\n{}\n{}}}",
                indent_s,
                dataset.axis_map.get_by_right(&c.axis).unwrap(),
                c.split,
                left,
                indent_s,
                right,
                indent_s
            )
        }
        DecisionTree::Leave(class) => {
            format!(
                "{}{}",
                indent_s,
                dataset.class_map.get_by_right(class).unwrap()
            )
        }
    }
}

fn assess(dataset: &DataSet, result: &HashMap<Class, Vec<DataPoint>>) -> f64 {
    let mut correct: usize = 0;
    let mut all: usize = 0;
    for (class, data) in result {
        for dp in data {
            if dataset.rows.iter().find(|row| row.0 == dp.0).unwrap().1
                == *dataset.class_map.get_by_right(class).unwrap()
            {
                correct += 1;
            }
            all += 1;
        }
    }
    correct as f64 / all as f64
}

fn log(log_ref: NodeRef<html::Custom>, line: &str) {
    let element = log_ref.get_untracked().unwrap().into_any().clone();
    let content = get_property(&element, "value")
        .unwrap_or_default()
        .as_string()
        .unwrap();
    set_property(&element, "value", &Some((content + line + "\n").into()));
}

macro_rules! log {
    ($e:expr, $($t:tt)*) => (log($e, &format!($($t)*)))
}

fn timestamp() -> f64 {
    window().performance().unwrap().now() as f64 / 1000.
}

#[component]
pub fn Main() -> impl IntoView {
    let (dataset, set_dataset) = create_signal(None::<DataSet>);
    let (train_data, set_train_data) = create_signal(None::<Vec<(DataPoint, Class)>>);
    let (train_classify_data, set_train_classify_data) = create_signal(None::<Vec<DataPoint>>);
    let (classify_data, set_classify_data) = create_signal(None::<Vec<DataPoint>>);
    let (tree, set_tree) = create_signal(None::<DecisionTree>);
    let (result, set_result) = create_signal(None::<DataSet>);
    let log_ref: NodeRef<html::Custom> = create_node_ref();
    let x_ref: NodeRef<html::Custom> = create_node_ref();
    let y_ref: NodeRef<html::Custom> = create_node_ref();
    create_effect(move |_| {
        let Some(dataset) = dataset() else {
            return;
        };
        let x = x_ref.get_untracked().unwrap();
        let y = y_ref.get_untracked().unwrap();
        let nodes: Vec<_> = dataset
            .axises
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let element = document().create_element("sl-option").unwrap();
                element.set_attribute("value", &i.to_string()).unwrap();
                element.set_text_content(Some(label));
                element
            })
            .collect();
        let element = Element::from((*x.into_any()).clone());
        element.replace_children_with_node(&Array::from_iter(nodes.clone()));
        set_property(&element, "value", &Some("".into()));
        let nodes: Vec<_> = dataset
            .axises
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let element = document().create_element("sl-option").unwrap();
                element.set_attribute("value", &i.to_string()).unwrap();
                element.set_text_content(Some(label));
                element
            })
            .collect();
        let element = Element::from((*y.into_any()).clone());
        element.replace_children_with_node(&Array::from_iter(nodes.clone()));
        set_property(&element, "value", &Some("".into()));
    });
    let (x_axis, set_x_axis) = create_signal(None::<usize>);
    let (y_axis, set_y_axis) = create_signal(None::<usize>);
    let chart_ref: NodeRef<html::Canvas> = create_node_ref();
    let dataset_not_found_alert: NodeRef<html::Custom> = create_node_ref();
    let x_y_same_alert: NodeRef<html::Custom> = create_node_ref();
    create_effect(move |previous_chart: Option<Option<JsValue>>| {
        let chart = chart_ref()?;
        if previous_chart.is_none() {
            x_axis.track();
            let chart = (*chart.into_any()).clone();
            let config = Object::new();
            set(&config, "type", &"scatter".into());
            set(
                &config,
                "options",
                &{
                    let obj = Object::new();
                    set(
                        &obj,
                        "scales",
                        &{
                            let obj = Object::new();
                            set(
                                &obj,
                                "x",
                                &{
                                    let obj = Object::new();
                                    set(&obj, "type", &"linear".into());
                                    set(&obj, "position", &"bottom".into());
                                    obj
                                }
                                .into(),
                            );
                            obj
                        }
                        .into(),
                    );
                    obj
                }
                .into(),
            );
            let chart = new_chart(&chart, &config);
            return Some(chart);
        };
        let previous_chart = previous_chart.unwrap();
        let Some(x_axis) = x_axis() else {
            return previous_chart;
        };
        let Some(y_axis) = y_axis() else {
            return previous_chart;
        };
        if x_axis == y_axis {
            toast_alert(x_y_same_alert);
            return previous_chart;
        }
        let previous_chart = previous_chart.unwrap();
        let dataset = dataset().unwrap();
        let data = into_chart_dataset(x_axis, y_axis, &dataset.classes, &dataset.rows);
        set(&previous_chart, "data", &data);
        update_chart(&previous_chart);
        Some(previous_chart)
    });
    let (class_name, style_val) = style_str! {
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            width: 50%;
            align-items: stretch;
            gap: 1rem;
            padding: 1rem;
        }
        h2 {
            margin: 0;
        }
        .controls {
            display: flex;
            flex-direction: row;
            align-items: end;
            gap: 1rem;
        }
        sl-textarea::part(textarea) {
            white-space: pre-wrap;
            overflow-wrap: break-word;
            font-family: "DejaVu Sans Mono", ui-monospace, "Cascadia Code", Menlo,
            "Source Code Pro", Consolas, monospace;
        }
    };
    view! {
        class = class_name,
        <Style> { style_val } </Style>
        <main>
            <h2> "决策树" </h2>
            <div class="controls">
                <sl-select label="选择数据集" on:sl-change=move |ev: JsValue| {
                    let begin = timestamp();
                    let source = get(&get(&ev, "target"), "value").as_string().unwrap();
                    set_x_axis(None);
                    set_y_axis(None);
                    set_tree(None);
                    let csv = load_csv(source);
                    let dataset = load_dataset(csv);
                    let (
                        train_data,
                        train_classify_data,
                        classify_data
                    ) = split_train_classify_dataset(dataset.clone());
                    set_dataset(Some(dataset));
                    let duration = (0.001f64).max(timestamp() - begin);
                    log!(log_ref, "数据集加载完毕, 训练集点数 {}, 测试集点数 {}, 用时 {duration:.3} 秒", train_data.len(), classify_data.len());
                    set_train_data(Some(train_data));
                    set_train_classify_data(Some(train_classify_data));
                    set_classify_data(Some(classify_data));
                }>
                    <sl-option value="iris"> "鸢尾花的尺寸" </sl-option>
                    <sl-option value="wine"> "红酒的理化性质" </sl-option>
                    <sl-option value="rice"> "米粒的形状" </sl-option>
                </sl-select>
                <sl-select label="X 轴数据点" ref=x_ref on:sl-change=move |ev: JsValue| {
                    match get(&get(&ev, "target"), "value").as_string().unwrap().parse().ok() {
                        Some(x) => set_x_axis(Some(x)),
                        None => set_x_axis(None),
                    }
                }>
                </sl-select>
                <sl-select label="Y 轴数据点" ref=y_ref on:sl-change=move |ev: JsValue| {
                    match get(&get(&ev, "target"), "value").as_string().unwrap().parse().ok() {
                        Some(y) => set_y_axis(Some(y)),
                        None => set_y_axis(None),
                    }
                }>
                </sl-select>
                <sl-button-group>
                    <sl-button on:click=move |_| {
                        let begin = timestamp();
                        let Some(dataset) = dataset() else {
                            toast_alert(dataset_not_found_alert);
                            return;
                        };
                        let tree = train(
                            dataset.mapped_axis.as_slice(),
                            dataset.mapped_class.as_slice(),
                            train_data().unwrap(),
                        );
                        let result = classify(&tree, train_classify_data().unwrap());
                        let correct_rate = assess(&dataset, &result);
                        log!(log_ref, "决策树:\n{}", display_tree(&dataset, &tree, 0));
                        set_tree(Some(tree));
                        let duration = (0.001f64).max(timestamp() - begin);
                        log!(log_ref, "训练完毕, 训练集分类正确率 {:.3}%, 用时 {duration:.3} 秒", correct_rate * 100.);
                    }> "训练" </sl-button>
                    <sl-button on:click=move |_| {
                        let begin = timestamp();
                        let Some(dataset) = dataset() else {
                            toast_alert(dataset_not_found_alert);
                            return;
                        };
                        let result = classify(&tree().unwrap(), classify_data().unwrap());
                        let correct_rate = assess(&dataset, &result);
                        let duration = (0.001f64).max(timestamp() - begin);
                        log!(log_ref, "分类完毕, 测试集分类正确率 {:.3}%, 用时 {duration:.3} 秒", correct_rate * 100.);
                    }> "分类" </sl-button>
                </sl-button-group>
                <sl-alert variant="danger" duration="3000" closable ref=dataset_not_found_alert>
                    <sl-icon slot="icon" name="exclamation-octagon"></sl-icon>
                    "必须选择数据集"
                </sl-alert>
                <sl-alert variant="danger" duration="3000" closable ref=x_y_same_alert>
                    <sl-icon slot="icon" name="exclamation-octagon"></sl-icon>
                    "X 轴和 Y 轴数据不能相同"
                </sl-alert>
            </div>
            <div class="chart">
                <canvas ref=chart_ref />
            </div>
            <sl-textarea label="日志" rows="15" ref=log_ref></sl-textarea>
        </main>
    }
}
