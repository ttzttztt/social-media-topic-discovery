"""
BERTopic 主题模型训练脚本 — 系统化调参版本

调参顺序（按影响力从大到小）：
  第1步：min_cluster_size  → 控制主题数量（最关键）
  第2步：n_neighbors       → 优化全局 vs 局部结构
  第3步：n_components      → 找到最佳降维维度
  第4步：停用词 + min_df   → 提升关键词可读性
  第5步：min_samples       → 微调离群点比例

用法：
  conda activate bertopic
  python train.py                    # 运行全部调参实验
  python train.py --mode quick       # 快速模式（只跑推荐配置）
  python train.py --mode single --name my_config  # 单配置
"""

import os, sys, io, json, re, time, argparse, warnings
from datetime import datetime
from pathlib import Path

import jieba
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ========================== 中文停用词 ==========================
# stopwords.txt — 哈工大 + 百度 + 川大 + 网络领域词，共 859 词

def _load_stopwords(filepath="stopwords.txt"):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []

STOPWORDS = _load_stopwords()

# ========================== 工具函数 ==========================

def load_docs(data_path="data.csv"):
    df = pd.read_csv(data_path)
    docs = df["content"].dropna().tolist()
    docs = [d.strip() for d in docs if len(str(d).strip()) >= 10]
    print(f"[数据] 加载 {len(docs)} 篇文档")
    return docs


def chinese_tokenizer(text):
    return [w for w in jieba.cut(text) if len(w.strip()) > 1]


def clean_topic_words(words):
    """清洗主题词：去标点符号和特殊字符，保留中文、英文、数字及全角形式"""
    cleaned = []
    for w in words:
        # CJK基本 + 扩展A + 全角数字 + 全角字母 + ASCII字母 + ASCII数字 + 下划线
        w = re.sub(
            r'[^一-鿿㐀-䶿'
            r'０-９Ａ-Ｚａ-ｚ'
            r'a-zA-Z0-9_]',
            '', w.strip()
        )
        if w:
            cleaned.append(w)
    return cleaned


# ========================== 模型构建 ==========================

def build_model(config):
    """根据配置构建 BERTopic 模型"""
    umap_model = UMAP(
        n_neighbors=config.get('n_neighbors', 30),
        n_components=config.get('n_components', 5),
        min_dist=config.get('min_dist', 0.0),
        metric='cosine',
        random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.get('min_cluster_size', 100),
        min_samples=config.get('min_samples', 5),
        metric='euclidean',
        prediction_data=True,
        cluster_selection_epsilon=config.get('cluster_selection_epsilon', 0.0),
    )
    vectorizer_model = CountVectorizer(
        tokenizer=chinese_tokenizer,
        max_features=config.get('max_features', 3000),
        ngram_range=config.get('ngram_range', (1, 2)),
        stop_words=STOPWORDS if config.get('use_stopwords', True) else None,
        min_df=config.get('min_df', 5),
    )

    topic_model = BERTopic(
        embedding_model=config.get('embedding_model_name', 'shibing624/text2vec-base-chinese'),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=15,
        verbose=True,
        language="chinese",
    )
    return topic_model


# ========================== 评估 ==========================

def evaluate(topic_model, docs, topics, embeddings):
    metrics = {}
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1  # exclude outlier topic (-1)
    n_outliers = sum(1 for t in topics if t == -1)
    metrics['n_topics'] = n_topics
    metrics['outlier_ratio'] = round(n_outliers / len(topics), 4)
    metrics['n_docs'] = len(docs)
    metrics['n_outliers'] = n_outliers

    # Coherence
    try:
        tokenized = [list(jieba.cut(doc)) for doc in docs]
        dictionary = Dictionary(tokenized)
        topic_words_list = []
        for tid in range(n_topics):
            words = [w for w, _ in topic_model.get_topic(tid)[:10]]
            cw = clean_topic_words(words)
            if cw:
                topic_words_list.append(cw)
        if topic_words_list:
            cm = CoherenceModel(
                topics=topic_words_list, texts=tokenized,
                dictionary=dictionary, coherence='c_v', processes=1
            )
            metrics['coherence'] = round(cm.get_coherence(), 4)
    except Exception as e:
        print(f"   ⚠ Coherence 计算失败: {e}")

    # Topic Diversity
    try:
        all_words = set()
        total_words = 0
        for tid in range(n_topics):
            words = [w for w, _ in topic_model.get_topic(tid)[:10]]
            all_words.update(words)
            total_words += len(words)
        metrics['diversity'] = round(len(all_words) / total_words, 4) if total_words else 0
    except:
        pass

    # Silhouette
    try:
        mask = np.array(topics) != -1
        unique_labels = set(np.array(topics)[mask])
        if mask.sum() >= 50 and len(unique_labels) >= 2:
            metrics['silhouette'] = round(
                silhouette_score(embeddings[mask], np.array(topics)[mask], metric='cosine'), 4
            )
    except:
        pass

    return metrics


# ========================== 结果保存 ==========================

def save_results(model, metrics, config, name):
    ts = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = Path(f"results/{name}_{ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(save_dir / "model"))
    with open(save_dir / "metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(save_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if callable(v) else v for k, v in config.items()},
                  f, ensure_ascii=False, indent=2)

    # 主题关键词
    keywords = {}
    for tid in range(metrics['n_topics']):
        words = [w for w, _ in model.get_topic(tid)[:15]]
        keywords[f"Topic_{tid}"] = clean_topic_words(words)
    with open(save_dir / "keywords.json", 'w', encoding='utf-8') as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)

    return save_dir


# ========================== 单次训练 ==========================

def run_one(docs, config, name, idx=None, total=None):
    prefix = f"[{idx}/{total}] " if idx else ""
    print(f"\n{'='*60}")
    print(f"{prefix}{name}")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    t0 = time.time()

    print(f"\n  [1/3] 生成嵌入向量 ({len(docs)} 篇)...")
    emb_model = SentenceTransformer(config.get('embedding_model_name', 'shibing624/text2vec-base-chinese'))
    embeddings = emb_model.encode(docs, show_progress_bar=True)

    print(f"  [2/3] 训练 BERTopic...")
    topic_model = build_model(config)
    topics, _ = topic_model.fit_transform(docs, embeddings)

    elapsed = time.time() - t0

    print(f"  [3/3] 评估...")
    metrics = evaluate(topic_model, docs, topics, embeddings)
    metrics['time_seconds'] = round(elapsed, 1)

    print(f"\n  ✅ 完成 ({elapsed:.0f}s)")
    print(f"  主题数: {metrics['n_topics']}  离群比例: {metrics['outlier_ratio']:.1%}  "
          f"Coherence: {metrics.get('coherence', 'N/A')}  多样性: {metrics.get('diversity', 'N/A')}  "
          f"轮廓系数: {metrics.get('silhouette', 'N/A')}")

    save_dir = save_results(topic_model, metrics, config, name)
    return metrics, save_dir, topic_model


# ========================== 配置定义 ==========================

def get_tuning_configs():
    """
    降低离群比例 — 基于 mcs_200 最优参数，只调离群相关杠杆。
    固定：n_neighbors=30, n_components=5, min_cluster_size=200
    变量：min_samples, cluster_selection_epsilon
    """
    configs = {}

    # 基准：mcs_200（复现）
    configs["baseline"] = {
        "embedding_model_name": "shibing624/text2vec-base-chinese",
        "n_neighbors": 30, "n_components": 5, "min_dist": 0.0,
        "min_cluster_size": 200, "min_samples": 5,
        "cluster_selection_epsilon": 0.0,
        "max_features": 3000, "ngram_range": (1, 2),
        "min_df": 5, "use_stopwords": True,
    }

    # min_samples：降低 → 更少点被判为噪声
    for ms in [1, 3]:
        configs[f"ms_{ms}"] = {
            "embedding_model_name": "shibing624/text2vec-base-chinese",
            "n_neighbors": 30, "n_components": 5, "min_dist": 0.0,
            "min_cluster_size": 200, "min_samples": ms,
            "cluster_selection_epsilon": 0.0,
            "max_features": 3000, "ngram_range": (1, 2),
            "min_df": 5, "use_stopwords": True,
        }

    # cluster_selection_epsilon：合并邻近聚类，覆盖噪声地带
    for eps in [0.1, 0.25]:
        configs[f"eps_{eps}"] = {
            "embedding_model_name": "shibing624/text2vec-base-chinese",
            "n_neighbors": 30, "n_components": 5, "min_dist": 0.0,
            "min_cluster_size": 200, "min_samples": 5,
            "cluster_selection_epsilon": eps,
            "max_features": 3000, "ngram_range": (1, 2),
            "min_df": 5, "use_stopwords": True,
        }

    # 组合
    configs["combo_ms3_eps01"] = {
        "embedding_model_name": "shibing624/text2vec-base-chinese",
        "n_neighbors": 30, "n_components": 5, "min_dist": 0.0,
        "min_cluster_size": 200, "min_samples": 3,
        "cluster_selection_epsilon": 0.1,
        "max_features": 3000, "ngram_range": (1, 2),
        "min_df": 5, "use_stopwords": True,
    }

    configs["combo_ms1_eps025"] = {
        "embedding_model_name": "shibing624/text2vec-base-chinese",
        "n_neighbors": 30, "n_components": 5, "min_dist": 0.0,
        "min_cluster_size": 200, "min_samples": 1,
        "cluster_selection_epsilon": 0.25,
        "max_features": 3000, "ngram_range": (1, 2),
        "min_df": 5, "use_stopwords": True,
    }

    return configs


def get_quick_configs():
    """快速模式：选 3 个代表性配置"""
    return {
        "quick_baseline": {
            "embedding_model_name": "shibing624/text2vec-base-chinese",
            "n_neighbors": 30, "n_components": 5, "min_dist": 0.0,
            "min_cluster_size": 200, "min_samples": 5,
            "cluster_selection_epsilon": 0.0,
            "max_features": 3000, "ngram_range": (1, 2),
            "min_df": 5, "use_stopwords": True,
        },
        "quick_ms_1": {
            "embedding_model_name": "shibing624/text2vec-base-chinese",
            "n_neighbors": 30, "n_components": 5, "min_dist": 0.0,
            "min_cluster_size": 200, "min_samples": 1,
            "cluster_selection_epsilon": 0.0,
            "max_features": 3000, "ngram_range": (1, 2),
            "min_df": 5, "use_stopwords": True,
        },
        "quick_eps_025": {
            "embedding_model_name": "shibing624/text2vec-base-chinese",
            "n_neighbors": 30, "n_components": 5, "min_dist": 0.0,
            "min_cluster_size": 200, "min_samples": 5,
            "cluster_selection_epsilon": 0.25,
            "max_features": 3000, "ngram_range": (1, 2),
            "min_df": 5, "use_stopwords": True,
        },
    }


# ========================== 结果输出 ==========================

def print_summary(all_results, configs):
    """打印对比表格并推荐最优配置"""
    print("\n\n" + "=" * 80)
    print("📊 调参结果汇总")
    print("=" * 80)

    rows = []
    for name in configs:
        r = all_results.get(name, {})
        rows.append({
            '配置': name,
            '主题数': r.get('n_topics', '-'),
            '离群%': f"{r.get('outlier_ratio', 0):.1%}" if r.get('outlier_ratio') else '-',
            'Coherence': r.get('coherence', '-'),
            '多样性': r.get('diversity', '-'),
            '轮廓系数': r.get('silhouette', '-'),
            '耗时': f"{r.get('time_seconds', 0):.0f}s",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    df.to_csv("data/tuning_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n📁 结果已保存: data/tuning_results.csv")

    # 推荐最优
    valid = [r for r in rows if isinstance(r.get('Coherence'), (int, float))]
    if valid:
        best = max(valid, key=lambda x: x['Coherence'])
        print(f"\n🏆 综合推荐: {best['配置']} (Coherence={best['Coherence']}, 主题数={best['主题数']})")
        print(f"   关键参数: {json.dumps({k: v for k, v in configs[best['配置']].items()}, ensure_ascii=False)}")

    return df


def inspect_best(model, docs, topics, n_samples=8):
    """抽样检查最优模型的主题质量"""
    print("\n\n" + "=" * 80)
    print("🔍 最优模型主题抽样检查")
    print("=" * 80)

    topic_info = model.get_topic_info()
    topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()

    import random
    sample_ids = random.sample(topic_ids, min(n_samples, len(topic_ids)))

    for tid in sorted(sample_ids):
        words = [w for w, _ in model.get_topic(tid)[:10]]
        print(f"\n  📌 主题 {tid}: {' | '.join(words)}")

        # 找该主题的代表文档
        idxs = [i for i, t in enumerate(topics) if t == tid]
        if idxs:
            for j in idxs[:2]:
                print(f"     └ {docs[j][:100]}...")


# ========================== 主入口 ==========================

def main():
    parser = argparse.ArgumentParser(description='BERTopic 主题模型训练')
    parser.add_argument('--mode', choices=['full', 'quick', 'single'], default='full',
                        help='full=全部调参  quick=3组快速对比  single=单配置')
    parser.add_argument('--name', default='step5_final',
                        help='single 模式下的配置名')
    parser.add_argument('--data', default='data.csv')
    parser.add_argument('--sample', type=int, default=0,
                        help='抽样 N 条数据（0=全部）')
    args = parser.parse_args()

    # 加载数据
    docs = load_docs(args.data)
    if args.sample > 0:
        docs = docs[:args.sample]
        print(f"[抽样] 取前 {len(docs)} 条")

    # 选择配置
    all_configs = get_tuning_configs()
    if args.mode == 'quick':
        all_configs = get_quick_configs()
    elif args.mode == 'single':
        all_configs = {args.name: all_configs.get(args.name, all_configs['step5_final'])}

    print(f"\n🚀 开始训练 — {len(all_configs)} 组配置")
    print(f"   文档数: {len(docs)}")

    all_results = {}
    best_model = None
    best_topics = None
    best_score = -1

    for idx, (name, config) in enumerate(all_configs.items(), 1):
        try:
            metrics, save_dir, model = run_one(docs, config, name, idx, len(all_configs))
            all_results[name] = metrics

            if metrics.get('coherence') and metrics['coherence'] > best_score:
                best_score = metrics['coherence']
                best_model = model
                best_topics = None  # Will need to recompute or store
        except Exception as e:
            print(f"\n  ❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    summary = print_summary(all_results, all_configs)

    # 抽样检查最优模型
    if len(all_configs) > 1:
        # Use the best from summary
        valid = [(r['配置'], r['Coherence']) for r in
                 [{'配置': n, **r} for n, r in all_results.items()]
                 if isinstance(r.get('coherence'), (int, float))]
        if valid:
            best_name = max(valid, key=lambda x: x[1])[0]
            print(f"\n🏆 最优配置: {best_name}")
            print(f"   查看关键词: results/{best_name}_*/keywords.json")
            print(f"   加载模型: from bertopic import BERTopic; m = BERTopic.load('results/{best_name}_*/model')")


if __name__ == "__main__":
    main()
