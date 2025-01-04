import inquirer

from few_shot_crew import (
    ReactionRule,
    ReactionTestSpecs,
    SimilarityRule,
    SimilarityTestSpecs,
    SingleSignalRule,
    SingleSignalTestSpecs,
    read_examples,
    run_crew,
)


def main():
    questions = [
        inquirer.List(
            "rule",
            message="Which kind of integrity check rule do you want to create?",
            choices=["Reaction Rule", "Single-Signal Rule", "Similarity Rule"],
        ),
    ]

    answers = inquirer.prompt(questions)

    examples = None
    empty_rule = None
    rule_type = None

    if answers["rule"] == "Reaction Rule":
        empty_rule = ReactionRule(
            test_method="",
            test_preconditions={},
            event_dict={},
            test_specs=ReactionTestSpecs(
                reaction_signal="",
                window_shift_event_start=0,
                window_shift_event_end=0,
                feature="",
                kind="",
                threshold=0,
                second_threshold=0,
            ),
            time_start_utc="",
            time_end_utc="",
            violation_plausibility_code="",
        )

        examples = read_examples("../examples/reaction")
        rule_type = ReactionRule
    elif answers["rule"] == "Single-Signal Rule":
        empty_rule = SingleSignalRule(
            test_method="",
            test_method_sub="",
            test_virtual_variables={},
            test_preconditions={},
            event_dict={},
            test_specs=SingleSignalTestSpecs(
                kind="",
                threshold=0,
                second_threshold=0,
                stat_feature="",
                window_length=0,
                min_periods=0,
            ),
            time_start_utc="",
            time_end_utc="",
            violation_plausibility_code="",
            violation_agg_func_runs="",
        )

        examples = read_examples("../examples/single_signal")
        rule_type = SingleSignalRule
    elif answers["rule"] == "Similarity Rule":
        empty_rule = SimilarityRule(
            test_method="",
            test_preconditions={},
            test_specs=SimilarityTestSpecs(
                kind="",
                threshold=0,
                second_threshold=0,
                stat_feature="",
                window_length=0,
                min_periods=0,
            ),
            time_start_utc="",
            time_end_utc="",
            violation_plausibility_code="",
            violation_agg_func_runs="",
        )

        examples = read_examples("../examples/similarity")
        rule_type = SimilarityRule

    # Example: Check if the room temperature decreases when the cooling coil valve is opened
    query = input("What should the rule check for?\n")
    result = run_crew(query, examples, empty_rule, rule_type)
    print(result.model_dump_json(indent=4, exclude_none=True))


if __name__ == "__main__":
    main()
