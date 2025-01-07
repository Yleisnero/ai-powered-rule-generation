import json
from few_shot_crew import (
    ReactionRule,
    ReactionTestSpecs,
    SimilarityRule,
    SimilarityTestSpecs,
    SingleSignalRule,
    SingleSignalTestSpecs,
    run_crew,
)
from few_shot_crew import read_examples
import csv
import datetime


def exact_match(output, expected_output):
    return output == expected_output


def same_value_score(output: dict, expected_output: dict, score=0, max_score=0):
    for key, value in expected_output.items():
        if type(value) == dict:
            if key in output:
                next_output = output[key]
            else:
                next_output = {}
            score, max_score = same_value_score(next_output, value, score, max_score)
            continue

        
        if output is not None and key in output:
            print(f"output is {output[key]} vs. {value}")
            if output[key] == value:
                score += 1
        else:
            print(f"output is missing {key}")
        max_score += 1

    return score, max_score


def evaluate(example, examples, filename, empty_rule, rule_type):
    evaluation_split = example
    examples_split = [ex for ex in examples if ex != example]
    input = evaluation_split["input"]
    expected_output = (
        evaluation_split["output"]
        .replace("{{", "{")
        .replace("}}", "}")
        .replace("'", '"')
    )

    try:
        output = run_crew(input, examples_split, empty_rule, rule_type)
        output = output.model_dump()
    except Exception as e:
        print("Error:", e)
        output = {}

    expected_output = json.loads(expected_output)

    print("Input:", input)
    print("Expected output:", expected_output)
    print("Output:", output)

    print("Exact match:", exact_match(output, expected_output))
    score, max_score = same_value_score(output, expected_output)
    print(f"Correct values: {score} / Total values: {max_score}")
    percentage_score = round((score / max_score) * 100, 2)
    print("Correctness score: " + str(percentage_score) + "%")
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [input, expected_output, output, score, max_score, percentage_score]
        )


def evaluate_all(examples, filename, empty_rule, rule_type):
    # TODO: run with different k for the examples
    # split: 1 evaluation, rest (16 reaction) as examples, top 4 examples are picked
    for example in examples:
        evaluate(example, examples, filename, empty_rule, rule_type)


def main():
    print("📖 Start generating a rule ...")
    empty_reaction_rule = ReactionRule(
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
            accepted_ratio_invalid=0,
        ),
        time_start_utc="",
        time_end_utc="",
        violation_plausibility_code="",
    )

    print("🔎 Looking for examples ...")
    examples = read_examples("../examples/reaction")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"evaluate_{timestamp}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input",
                "expected_output",
                "output",
                "score",
                "max_score",
                "percentage_score",
            ],
        )
        writer.writeheader()

    evaluate_all(examples, filename, empty_reaction_rule, ReactionRule)

    empty_single_signal_rule = SingleSignalRule(
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

    examples_single_signal = read_examples("../examples/single_signal")

    evaluate_all(examples_single_signal, filename, empty_single_signal_rule, SingleSignalRule)

    empty_similarity_rule = SimilarityRule(
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

    examples_similarity = read_examples("../examples/similarity")

    evaluate_all(examples_similarity, filename, empty_similarity_rule, SimilarityRule)


if __name__ == "__main__":
    main()
