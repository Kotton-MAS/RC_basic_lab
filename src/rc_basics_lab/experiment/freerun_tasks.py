"""実験04 の課題の組み立てと、区間の検査.

``experiment/freerun.py`` から**課題側だけ**を切り出したモジュール。
ここが持つのは「どの課題を、どの長さで、どこを標準化して作るか」だけで、
教師強制も自走も知らない。

**標準化の窓は訓練区間の内側でなければならない** (D-41)。ここを外すと、
テスト区間の情報が係数に漏れる。``validate_standardization_window`` が
分割と突き合わせて落とす。
"""

from __future__ import annotations

from rc_basics_lab.config import (
    Chaos04Config,
    ESNConfig,
    ExperimentConfig,
    MackeyGlassTask,
    require_task,
)
from rc_basics_lab.experiment.capacity_bounds import validate_state_matrix_bounds
from rc_basics_lab.experiment.runner import TaskEntry
from rc_basics_lab.experiment.split import Split
from rc_basics_lab.reservoir.registry import require_esn
from rc_basics_lab.tasks.chaotic import (
    TASK_NAME_LORENZ,
    generate_lorenz,
    generate_standardized_mackey_glass,
    sampling_interval,
)
from rc_basics_lab.tasks.mackey_glass import TASK_NAME as TASK_NAME_MACKEY_GLASS


def chaos_esn_config(base: ExperimentConfig) -> ESNConfig:
    """04 が使う ESN 設定を返す (``CHAOS_ESN_SECTION`` の1本)。

    「どのセクションを読むか」をここ1か所に閉じる。呼び出し側が属性を直接
    書くと、宣言したセクションと実際に読むセクションが食い違っても何も落ちない。
    """
    task = require_task(base, MackeyGlassTask, "実験04 (カオス時系列の自由走行)")
    return require_esn(task.reservoir, "実験04 (カオス時系列の自由走行)")


def lorenz_task_entry(config: Chaos04Config) -> TaskEntry:
    """Lorenz の ``TaskEntry`` を組む (**``build_tasks`` には足さない**、D-31)。"""
    return TaskEntry(
        name=TASK_NAME_LORENZ,
        reservoir=chaos_esn_config(config.base),
        generate=lambda rng: generate_lorenz(config.lorenz, rng),
    )


def mackey_glass_task_entry(config: Chaos04Config) -> TaskEntry:
    """04 の MG の ``TaskEntry`` を組む (生成は 01 の実装へ委譲、D-41)。

    生成パラメータの単一の真実は ``config.base.mackey_glass`` (01 の
    ``MackeyGlassConfig``) で、04 が足すのは標準化だけである。
    """
    return TaskEntry(
        name=TASK_NAME_MACKEY_GLASS,
        reservoir=chaos_esn_config(config.base),
        generate=lambda rng: generate_standardized_mackey_glass(
            require_task(config.base, MackeyGlassTask, "実験04 の MG 課題").params,
            rng,
            standardize_steps=config.mackey_glass.standardize_steps,
        ),
    )


def chaos_task_entries(config: Chaos04Config) -> tuple[TaskEntry, ...]:
    """04 が回す課題 (Lorenz 主 + MG 従、仕様 §8)。**課題を列挙する唯一の場所**。"""
    return (lorenz_task_entry(config), mackey_glass_task_entry(config))


def task_length(config: Chaos04Config, task_name: str) -> int:
    """課題名 -> 系列長。確保軸の検査で「何行の状態を作るか」を知るために使う。

    ``TASK_LENGTH_FIELDS`` (01) と同じ役割だが、04 は Lorenz の長さを
    ``config.lorenz`` に、MG の長さを ``config.base.mackey_glass`` に持つので
    対応表がここに要る。未知の課題名は ``ValueError`` にする (課題を足して
    ここへの登録を忘れると確保軸の検査が黙って効かなくなる)。
    """
    match task_name:
        case _ if task_name == TASK_NAME_LORENZ:
            return config.lorenz.length
        case _ if task_name == TASK_NAME_MACKEY_GLASS:
            return require_task(
                config.base, MackeyGlassTask, "実験04 の系列長"
            ).params.length
        case _:
            raise ValueError(f"04 の課題ではありません: {task_name!r}")


def validate_free_run_bounds(free_run_steps: int, n_units: int) -> None:
    """確保軸3 (``free_run_steps * n_units``) を**確保より前に**検査する (D-34)。

    自走は ``(free_run_steps, n_units)`` の状態行列を確保するので、容量実験の
    状態行列とまったく同じ軸である。**04 で新しい上限を作らず**、
    ``experiment/capacity.py`` の ``validate_state_matrix_bounds`` を再利用する
    (上限が2か所にあると片方だけ緩められる)。

    Raises:
        ValueError: ``free_run_steps`` が 1 未満、または確保軸が上限を超える場合。
    """
    if free_run_steps < 1:
        raise ValueError(
            f"free_run_steps は 1 以上である必要があります: {free_run_steps}"
        )
    validate_state_matrix_bounds(n_units, free_run_steps)


def validate_standardization_window(standardize_steps: int, split: Split) -> None:
    """標準化係数の推定区間が訓練区間の内側に収まることを検査する (D-41)。

    ``Standardizer.from_training_prefix`` は系列の**先頭**から係数を推定する。
    先頭 ``standardize_steps`` 行が検証区間・テスト区間に食い込むと、評価区間の
    統計量が係数へ混ざる —— 予測が当たっていない区間でも平均・分散が揃うため
    「当たっているように見える」壊れ方をし、図でも有効予測時間でも検出できない
    (仕様 §10-2)。分割が決まるのは課題生成の**後**なので、検査はここで行う。

    Raises:
        ValueError: 推定区間が訓練区間の終端を超える場合。
    """
    if standardize_steps > split.train.stop:
        raise ValueError(
            "標準化係数の推定区間が訓練区間を越えています (D-41): "
            f"standardize_steps={standardize_steps} > train.stop={split.train.stop} "
            "(評価区間の統計量が係数に混ざると『当たっているように見える』"
            "壊れ方をする)"
        )


def task_sampling_interval(config: Chaos04Config, task_name: str) -> float:
    """課題名 -> サンプリング間隔 Delta t [時間]。

    Lorenz は 0.01 (``rk4_step`` 0.002 x ``sample_interval`` 5)、Mackey-Glass は
    1.0 (0.1 x 10) で**2桁違う**。有効予測時間を時間の単位で報告するときも、
    パワースペクトルの周波数軸を作るときもこの値で割るので、片方の Delta t を
    両方に使うと 100 倍ずれた量を同じ列に書くことになる。

    Raises:
        ValueError: 04 の課題でない場合。
    """
    match task_name:
        case _ if task_name == TASK_NAME_LORENZ:
            return sampling_interval(config.lorenz)
        case _ if task_name == TASK_NAME_MACKEY_GLASS:
            mackey_glass = require_task(
                config.base, MackeyGlassTask, "実験04 のサンプリング間隔"
            ).params
            return mackey_glass.rk4_step * mackey_glass.sample_interval
        case _:
            raise ValueError(f"04 の課題ではありません: {task_name!r}")


def standardize_steps_for(config: Chaos04Config, task_name: str) -> int:
    """課題名 -> 標準化係数の推定に使う先頭サンプル数 (D-41)。"""
    match task_name:
        case _ if task_name == TASK_NAME_LORENZ:
            return config.lorenz.standardize_steps
        case _ if task_name == TASK_NAME_MACKEY_GLASS:
            return config.mackey_glass.standardize_steps
        case _:
            raise ValueError(f"04 の課題ではありません: {task_name!r}")


__all__ = [
    "chaos_esn_config",
    "chaos_task_entries",
    "lorenz_task_entry",
    "mackey_glass_task_entry",
    "standardize_steps_for",
    "task_length",
    "task_sampling_interval",
    "validate_free_run_bounds",
    "validate_standardization_window",
]
