# pubtator_inference/app.py

import json
import os
import logging
import warnings
import torch  # for clearing GPU cache
from flask import Flask, render_template, request, redirect, url_for, Response, stream_with_context
from apscheduler.schedulers.background import BackgroundScheduler

from .config import FULLTEXT_DIR, PMID_LIST_FILE, Hours, Minutes
from .pub_inference import do_inference_for_variant
from .parser_utils import sanitize_filename
from .file_utils import load_all_pmids
from .lime_interpret_sentences import highlight_lime_in_paragraphs
from .predict import setup_inference, predict_classification, ner_inference
from .ner_entity import ner_bp
from .auto_update import start_scheduler
from transformers import BertTokenizer

# suppress user warnings from transformers, etc.
warnings.filterwarnings("ignore", category=UserWarning)

# basic logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# initialize Flask and register NER blueprint
app = Flask(__name__)
app.register_blueprint(ner_bp)  # mounts /ner_entity routes

# load classification model and tokenizer once at startup
CONFIG, CLASS_MODEL, ID2LABEL, _, DEVICE = setup_inference()
TOKENIZER = BertTokenizer.from_pretrained(CONFIG['data']['tokenizer_name'])
partial_tpl = app.jinja_env.get_template("partial_results.html")

# schedule daily auto-update of all cached variants
scheduler = start_scheduler()


@app.route("/")
def index():
    """Home page: list all variants we've ever fetched."""
    variants = list(load_all_pmids(PMID_LIST_FILE).keys())
    return render_template("index.html", variants=variants)


@app.route("/result", methods=["POST"])
def result():
    """Handle the basic 'Search Variant' form (no inference)."""
    variant = request.form.get("variant", "").strip()
    if not variant:
        return redirect(url_for("index"))
    try:
        data, _ = do_inference_for_variant(
            variant,
            base_output_dir=FULLTEXT_DIR,
            pmid_list_file=PMID_LIST_FILE
        )
        if not data:
            return render_template("result.html", variant=variant,
                                   variant_data=None,
                                   error="No related data found.")
        return render_template("result.html", variant=variant, variant_data=data)
    except Exception as e:
        return render_template("result.html", variant=variant,
                               variant_data=None, error=str(e))


@app.route("/variant/<variant>")
def variant_view(variant):
    """Show list of PMIDs & titles for a saved variant."""
    all_pmids = load_all_pmids(PMID_LIST_FILE)
    if variant not in all_pmids:
        return render_template("index.html", alert="No data for this variant.",
                               variants=list(all_pmids.keys()))
    san = sanitize_filename(variant)
    path = os.path.join(FULLTEXT_DIR, f"{san}.json")
    if not os.path.exists(path):
        return render_template("index.html", alert="Data file missing.",
                               variants=list(all_pmids.keys()))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    articles = [{"pmid": pmid, "title": content.get("Title", "No Title")}
                for pmid, content in data.items()]
    return render_template("variant.html", variant=variant, articles=articles)


@app.route("/article/<variant>/<pmid>")
def article(variant, pmid):
    """Show the full text sections for a single PMID."""
    san = sanitize_filename(variant)
    path = os.path.join(FULLTEXT_DIR, f"{san}.json")
    if not os.path.exists(path):
        return render_template("index.html", alert="No data file.",
                               variants=list(load_all_pmids(PMID_LIST_FILE).keys()))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    content = data.get(pmid)
    if not content:
        return render_template("index.html", alert="PMID not found.",
                               variants=list(load_all_pmids(PMID_LIST_FILE).keys()))
    return render_template("article.html", variant=variant,
                           pmid=pmid, content=content)


@app.route("/inference_page", methods=["GET", "POST"])
def inference_page():
    variants = list(load_all_pmids(PMID_LIST_FILE).keys())

    if request.method == "POST":
        text = request.form.get("inference_text", "").strip()
        num_samples = int(request.form.get("num_samples", 300) or 300)

        # 1) 空输入就报错
        if not text:
            return render_template("inference.html",
                                   variants=variants,
                                   error="Please enter the paragraph containing the genetic variant.",
                                   num_samples=num_samples)

        paras = [p for p in text.split("\n") if p.strip()]

        # 2) 先做 NER 预筛，没有 variant 就直接报错
        if not ner_inference(paras):
            return render_template("inference.html",
                                   variants=variants,
                                   error="Please enter the correct paragraph related to genetic variant.",
                                   inference_text=text,
                                   num_samples=num_samples)

        # 3) 分类预测
        preds = predict_classification([paras], CONFIG, CLASS_MODEL, ID2LABEL)
        prediction = (f"[Classification Result] Predicted: {preds[0]}"
                      if preds else "[No prediction]")

        # 4) LIME 解释
        lime_html = highlight_lime_in_paragraphs(
            paragraphs=paras,
            model=CLASS_MODEL,
            tokenizer=TOKENIZER,
            class_names=["benign", "pathogenic"],
            device=DEVICE,
            base_threshold=0.1,
            num_samples=num_samples
        )

        return render_template("inference.html",
                               variants=variants,
                               prediction=prediction,
                               inference_text=text,
                               lime_html=lime_html,
                               num_samples=num_samples)

    # GET 请求
    return render_template("inference.html",
                           variants=variants,
                           num_samples=300)


@app.route("/search_inference", methods=["GET"])
def search_inference():
    """Page with SSE-powered 'Query Variant + Inference' form."""
    variants = list(load_all_pmids(PMID_LIST_FILE).keys())
    return render_template("search_inference.html", variants=variants)


@app.route("/search_inference_stream")
def search_inference_stream():
    """
    SSE endpoint: for a given variant, fetch (or load cache),
    extract paragraphs, classify & LIME-highlight them one by one.
    """
    variant = request.args.get("variant", "").strip()
    num_samples = int(request.args.get("num_samples", 300) or 300)
    if not variant:
        return Response(status=204)

    san = sanitize_filename(variant)
    json_path = os.path.join(FULLTEXT_DIR, f"{san}.json")

    # 1) Try local cache first
    if os.path.exists(json_path):
        logger.info(f"🔍 Cache hit, loading {json_path}")
        with open(json_path, encoding="utf-8") as f:
            variant_data = json.load(f)
    else:
        logger.info(f"🌐 Cache miss, fetching variant={variant}")
        variant_data, _ = do_inference_for_variant(
            variant,
            base_output_dir=FULLTEXT_DIR,
            pmid_list_file=PMID_LIST_FILE
        )
        # no results at all
        if not variant_data:
            def error_sse():
                yield "data: " + json.dumps({
                    "step":    "error",
                    "message": "No PMID data found"
                }) + "\n\n"
            return Response(stream_with_context(error_sse()),
                            mimetype="text/event-stream")

    # 2) Extract only paragraphs containing the variant
    extracted = []
    for pmid, content in variant_data.items():
        paras = []
        for section, text in content.items():
            if section.lower() in ("title", "pubmed_link"):
                continue
            for line in text.split("\n"):
                if variant in line:
                    paras.append(line.strip())
        if paras:
            extracted.append((pmid, paras, content.get("Title", "No Title")))

    # no matching paragraphs
    if not extracted:
        def nomatch_sse():
            yield "data: " + json.dumps({
                "step":    "error",
                "message": "No PMID data found"
            }) + "\n\n"
        return Response(stream_with_context(nomatch_sse()),
                        mimetype="text/event-stream")

    # 3) Batch classify
    all_paras = [paras for (_pmid, paras, _title) in extracted]
    all_preds = predict_classification(all_paras,
                                       CONFIG, CLASS_MODEL, ID2LABEL)

    # 4) Stream LIME‐highlighted results one by one
    def generate():
        count = 0
        try:
            for (pmid, paras, title), pred in zip(extracted, all_preds):
                lime_html = highlight_lime_in_paragraphs(
                    paragraphs=paras,
                    model=CLASS_MODEL,
                    tokenizer=TOKENIZER,
                    class_names=["benign", "pathogenic"],
                    device=DEVICE,
                    base_threshold=0.1,
                    num_samples=num_samples
                )
                fragment = partial_tpl.render(results=[{
                    "pmid":       pmid,
                    "title":      title,
                    "prediction": pred,
                    "lime_html":  lime_html
                }])
                count += 1
                yield "data: " + json.dumps({
                    "step":  "result",
                    "count": count,
                    "html":  fragment
                }) + "\n\n"
        except GeneratorExit:
            logger.info("⚠️ Client disconnected, stopping generation early")
        finally:
            # always clear GPU cache on exit
            torch.cuda.empty_cache()

        # done signal (if loop completes)
        yield "data: " + json.dumps({
            "step":  "done",
            "total": count
        }) + "\n\n"

    resp = Response(stream_with_context(generate()),
                    mimetype="text/event-stream")
    # clear GPU also if client forcibly closes
    resp.call_on_close(lambda: torch.cuda.empty_cache())
    return resp


if __name__ == "__main__":
    # disable Flask reloader to avoid scheduler duplication
    app.run(host="0.0.0.0", port=8080,
            debug=False, use_reloader=False)
