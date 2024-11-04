use std::collections::HashSet;

use js_sys::{Array, Object, Reflect};
use leptos::logging::log;
use leptos::*;
use leptos_dom::helpers::set_property;
use leptos_meta::*;
use stylers::style_str;
use wasm_bindgen::{prelude::*, JsValue};
use web_sys::Element;

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

#[component]
pub fn Button(#[prop(default = 1)] increment: i32) -> impl IntoView {
    let (count, set_count) = create_signal(0);
    view! {
        <button
            on:click= move |_| {
                set_count(count() + increment)
            }
        >
            "Click me: " {count}
        </button>
    }
}

#[derive(Debug, Clone)]
struct DataSet {
    axises: Vec<String>,
    classes: Vec<String>,
    rows: Vec<(Vec<f64>, String)>,
}

fn load_data(source: String, set_data: WriteSignal<Option<DataSet>>) {
    let csv = match source.as_str() {
        "iris" => {
            include_str!("../public/iris.data")
        }
        "dataset-2" => "",
        "dataset-3" => "",
        "custom" => "",
        _ => unreachable!(),
    };
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv.as_bytes());
    let mut axises: Vec<String> = rdr.headers().unwrap().iter().map(str::to_owned).collect();
    axises.pop().unwrap();
    let mut classes = HashSet::new();
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
    set_data(Some(DataSet {
        axises,
        classes: classes.into_iter().collect(),
        rows,
    }))
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

#[wasm_bindgen(inline_js = "export function toast_alert(alert) { alert.toast(); }")]
extern "C" {
    fn toast_alert(alert: &JsValue);
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

#[component]
pub fn Main() -> impl IntoView {
    let (data, set_data) = create_signal(None::<DataSet>);
    let x_ref: NodeRef<html::Custom> = create_node_ref();
    let y_ref: NodeRef<html::Custom> = create_node_ref();
    create_effect(move |_| {
        let Some(data) = data() else {
            return;
        };
        let x = x_ref.get_untracked().unwrap();
        let y = y_ref.get_untracked().unwrap();
        let nodes: Vec<_> = data
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
        let nodes: Vec<_> = data
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
    let x_y_same_alert: NodeRef<html::Custom> = create_node_ref();
    create_effect(move |previous_chart: Option<Option<JsValue>>| {
        let chart = chart_ref.get_untracked()?;
        let Some(x_axis) = x_axis() else {
            return match previous_chart {
                Some(Some(previous_chart)) => Some(previous_chart),
                _ => None,
            };
        };
        let Some(y_axis) = y_axis() else {
            return match previous_chart {
                Some(Some(previous_chart)) => Some(previous_chart),
                _ => None,
            };
        };
        if x_axis == y_axis {
            toast_alert(&(x_y_same_alert.get_untracked().unwrap().into_any()));
            return match previous_chart {
                Some(Some(previous_chart)) => Some(previous_chart),
                _ => None,
            };
        }
        let data = data().unwrap();
        let data = into_chart_dataset(x_axis, y_axis, &data.classes, &data.rows);
        if let Some(Some(previous_chart)) = previous_chart {
            set(&previous_chart, "data", &data);
            update_chart(&previous_chart);
            Some(previous_chart)
        } else {
            let chart = (*chart.into_any()).clone();
            let config = Object::new();
            set(&config, "type", &"scatter".into());
            set(&config, "data", &data);
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
            Some(chart)
        }
    });
    let (class_name, style_val) = style_str! {
        main {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
            padding: 3rem;
        }
        .controls {
            display: flex;
            flex-direction: row;
            align-items: end;
            gap: 1rem;
        }
        .chart {
            width: 50%;
        }
    };
    view! {
        class = class_name,
        <Style> { style_val } </Style>
        <main>
            <div class="controls">
                <sl-select label="选择数据集" on:sl-change=move |ev: JsValue| {
                    let source = get(&get(&ev, "target"), "value").as_string().unwrap();
                    load_data(source, set_data);
                }>
                    <sl-option value="iris"> "鸢尾花 (Iris)" </sl-option>
                    <sl-option value="dataset-2"> "数据集 2" </sl-option>
                    <sl-option value="dataset-3"> "数据集 3" </sl-option>
                    <sl-option value="custom"> "上传自定义数据集" </sl-option>
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
                        log!("train");
                    }> "训练" </sl-button>
                    <sl-button on:click=move |_| {
                        log!("classify");
                    }> "拟合" </sl-button>
                </sl-button-group>
                <sl-alert variant="danger" duration="3000" closable ref=x_y_same_alert>
                    <sl-icon slot="icon" name="exclamation-octagon"></sl-icon>
                    "X 轴和 Y 轴数据不能相同"
                </sl-alert>
            </div>
            <div class="chart">
                <canvas ref=chart_ref />
            </div>
        </main>
    }
}
