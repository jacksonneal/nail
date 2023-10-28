from pandas import DataFrame

from .data import ERA_COL, TARGET_COL, read_features, read_training_data


def get_feature_corrs(
    df: DataFrame,
    group_by: str,
    features: list[str],
    corr_with: str,
) -> DataFrame:
    return df.groupby(group_by).apply(lambda g: g[features].corrwith(g[corr_with]))


def get_training_feature_corrs_by_era_with_target() -> DataFrame:
    return get_feature_corrs(read_training_data(), ERA_COL, read_features(), TARGET_COL)


def get_volatile_featues(n: int = 10) -> list[str]:
    corrs = get_training_feature_corrs_by_era_with_target()
    all_eras = corrs.index.sort_values()

    h1_eras = all_eras[: len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2 :]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    return corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()


if __name__ == "__main__":
    print(get_volatile_featues())
