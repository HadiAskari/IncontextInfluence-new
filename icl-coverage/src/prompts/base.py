import attr
import typing
import evaluate

common_templates = {
    'v1': {
        "train": """
{input_label}: {source}
{output_label}: {target}""".lstrip(),
        'test': """
{input_label}: {source}
{output_label}: """.lstrip()
    },

    'v2': {
        'train': "{source}\t{target}",
        'test': "{source}\t"
    },

    'with_context': {
        'train': """
{context_label}: {context}
{input_label}: {source}
{output_label}: {target}""".lstrip(),
        'test': """
{context_label}: {context}
{input_label}: {source}
{output_label}: """.lstrip()
    }
}

class ExampleTemplate:
    input_variables: list[str]

    def get_source(self, **kwargs):
        raise NotImplementedError

    def get_target(self, **kwargs):
        return NotImplementedError

    def format(self, test=False, embedding=False, **kwargs):
        raise NotImplementedError

    def parse_output(self, lm_output: str, **kwargs) -> str:
        raise lm_output.strip()

    def check_output(self, pred, actual_target, **kwargs):
        is_correct = pred == actual_target
        return dict(accuracy=is_correct * 100,)

    def undo_format(self, string):
        raise NotImplementedError


@attr.s(auto_attribs=True)
class GenerationExampleTemplate(ExampleTemplate):
    input_variables: list[str] = ['input', 'output']
    input_label: str = 'Input'
    output_label: str = 'Output'
    version: str = 'v1'

    def get_source(self, **kwargs):
        return kwargs[self.input_variables[0]].strip()

    def get_target(self, **kwargs):
        return kwargs[self.input_variables[1]].strip()

    @property
    def templates(self) -> dict[str, str]:
        return common_templates[self.version]

    def format(self, test=False, embedding=False, **kwargs):
        source = self.get_source(**kwargs)
        target = self.get_target(**kwargs)
        if embedding: return source
        template = self.templates['test'] if test else self.templates['train']
        return template.format(
            source=source, target=target,
            input_label=self.input_label,
            output_label=self.output_label)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        if self.version == 'v2':
            return lm_output
        elif self.version == 'v1':
            if lm_output.startswith(f'{self.output_label}: '):
                # output label needs to be stripped
                # eg. with chat LMs like TURBO
                return lm_output[len(f'{self.output_label}: '):].strip()
            else:
                return lm_output
        else:
            raise NotImplementedError

    def undo_format(self, string):
        # NOTE: this would fail if the source/target themselves contain the separatorS
        if self.version == 'v2':
            source = string[:string.find('\t')]
            target = string[string.find('\t') + 1:]
        else:
            source = string[:string.find('\n')]
            target = string[string.find('\n') + 1:]
            source = source[len(f'{self.input_label}: '):]
            target = target[len(f'{self.output_label}: '):]
            # source = source[source.find(': ') + len(': '):]
            # target = target[target.find(': ') + len(': '):]
        return dict(source=source, target=target)

@attr.s(auto_attribs=True)
class ClassificationExampleTemplate(GenerationExampleTemplate):
    choices: list[str] | dict[typing.Any, str]
    input_variables: list[str] = ['text', 'label']
    input_label: str = 'Text'
    output_label: str = 'Label'
    version: str = 'v1'

    def get_target(self, **kwargs):
        """ target for the LM to predict """
        label = kwargs[self.input_variables[-1]]
        if isinstance(self.choices, list) and isinstance(label, int):
            label = self.choices[label]
        elif isinstance(self.choices, dict) and label in self.choices:
            label = self.choices[label]
        return label

    def get_choices(self, **kwargs):
        """
        target candidates
        - classes in case of classification
        - choices in case of MCQ questions
        """
        if isinstance(self.choices, list):
            return self.choices
        elif isinstance(self.choices, dict):
            return list(self.choices.values())
        else:
            raise NotImplementedError

    def check_output(self, pred, **kwargs):
        target = self.get_target(**kwargs)
        is_correct = pred == target
        return dict(accuracy=is_correct * 100,)
    
    # def get_new_choices(self, **kwargs):
    #     return [choice.strip() for choice in kwargs[self.input_variables[1]]]

class ContextMixin:
    input_variables: list[str] = ['input', 'context', 'output']
    context_label: str = 'Context'
    version: str = 'with_context'
    embed_context: bool = False

    def get_context(self, **kwargs):
        return kwargs[self.input_variables[1]].strip()

    @property
    def template(self):
        assert self.version == 'with_context', 'ContextualizedClassificationExampleTemplate only works with prompt version="with_context"'
        return common_templates[self.version]

    def get_embedding_text(self, source, context):
        return source if not self.embed_context else f'{context}\n{source}'

    def format(self, test=False, embedding=False, **kwargs):
        source = self.get_source(**kwargs)
        context = self.get_context(**kwargs)
        target = self.get_target(**kwargs)
        if embedding:
            return self.get_embedding_text(source, context)
        template = self.templates['test'] if test else self.templates['train']
        return template.format(
            source=source, context=context, target=target,
            input_label=self.input_label,
            context_label=self.context_label,
            output_label=self.output_label
        )

@attr.s(auto_attribs=True)
class ContextualizedGenerationExampleTemplate(ContextMixin, GenerationExampleTemplate):
    input_variables: list[str] = ['input', 'context', 'output']
    context_label: str = 'Context'
    version: str = 'with_context'
    embed_context: bool = False

@attr.s(auto_attribs=True)
class ContextualizedClassificationExampleTemplate(ContextMixin, ClassificationExampleTemplate):
    input_variables: list[str] = ['text', 'context', 'label']
    context_label: str = 'Context'
    version: str = 'with_context'
    embed_context: bool = False
    
    # def get_new_choices(self, **kwargs):
    #     return [choice.strip() for choice in kwargs[self.input_variables[1]]]

semparse_prefix = "Translate the sentence into a logical form."
@attr.s(auto_attribs=True)
class SemparseExampleTemplate(GenerationExampleTemplate):
    input_variables: list[str] = ['source', 'target']
    input_label: str = 'Sentence'
    output_label: str = 'Logical Form'

break_prefix = "Decompose the sentence into a sequence of steps."
class BreakEvaluator():
    def __init__(self):
        import sys
        sys.path.append("third_party/qdecomp_with_dependency_graphs")
        from dependencies_graph.evaluation.logical_form_matcher import LogicalFromStructuralMatcher
        from dependencies_graph.evaluation.qdmr_to_logical_form_tokens import \
            QDMRToQDMRStepTokensConverter
        from evaluation.normal_form.normalized_graph_matcher import \
            NormalizedGraphMatchScorer
        from scripts.eval.evaluate_predictions import format_qdmr
        self.converter = QDMRToQDMRStepTokensConverter()
        self.matcher = LogicalFromStructuralMatcher()
        self.scorer = NormalizedGraphMatchScorer()
        self.format_qdmr = format_qdmr
        sys.path.pop()
    def lfem(self, x):
        try:
            question, generated, decomposition, index = x['question_text'], x['pred'], x['actual_target'], x['question_id']
            gold = self.format_qdmr(decomposition)
            pred = self.format_qdmr(generated)
            decomp_lf = self.converter.convert(question_id=str(index), question_text=question,
                                                decomposition=pred.to_break_standard_string())
            gold_lf = self.converter.convert(question_id=str(index), question_text=question,
                                                decomposition=gold.to_break_standard_string())
            s = self.matcher.is_match(question_id=str(index), question_text=question, graph1=decomp_lf,
                                        graph2=gold_lf)
            return s
        except Exception as e:
            return False

@attr.s(auto_attribs=True)
class BreakExampleTemplate(GenerationExampleTemplate):
    input_variables: list[str] = ['question_text', 'decomposition']
    input_label: str = 'Sentence'
    output_label: str = 'Decomposition'
    evaluator: BreakEvaluator = attr.field(factory=BreakEvaluator)

    def get_target(self, **kwargs):
        return kwargs[self.input_variables[1]].strip().replace('  ', ' ')

    def check_output(self, pred, actual_target, **kwargs):
        is_correct = pred == actual_target
        lfem = self.evaluator.lfem(kwargs | dict(pred=pred, actual_target=actual_target))
        return dict(accuracy=is_correct * 100, lfem=lfem * 100)

@attr.s(auto_attribs=True)
class NL2BashExampleTemplate(GenerationExampleTemplate):
    input_variables: list[str] = ['nl', 'bash']
    input_label: str = 'Sentence'
    output_label: str = 'Bash'
    bleu: typing.Any = attr.field(factory=lambda: evaluate.load('bleu'))

    def check_output(self, pred, actual_target, **kwargs):
        is_correct = pred == actual_target
        bleu = self.bleu.compute(predictions=[list(pred)], references=[[list(target)]])['bleu']
        return dict(accuracy=is_correct * 100, bleu=bleu * 100)


@attr.s(auto_attribs=True)
class MRCExampleTemplate(ContextualizedGenerationExampleTemplate):
    input_variables: list[str] = ['question', 'passage', 'answer']
    context_label: str = 'Question'
    input_label: str = 'Passage'
    output_label: str = 'Answer'

class DropExampleTemplate(MRCExampleTemplate):
    input_variables: list[str] = ['passage', 'question', 'answer_text']

@attr.s(auto_attribs=True)
class RTEExampleTemplate(ContextualizedClassificationExampleTemplate):
    choices: list[str] = ["True", "False"]
    input_variables: list[str] = ['hypothesis', 'premise', 'label']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train="""
{context}
Question: {source} True or False?
Answer: {target}""".lstrip(),
            test="""
{context}
Question: {source} True or False?
Answer: """.lstrip()
    )

@attr.s(auto_attribs=True)
class QNLIExampleTemplate(ContextualizedClassificationExampleTemplate):
    choices: list[str] = ["Yes", "No"]
    input_variables: list[str] = ['question', 'sentence', 'label']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='{context} Can we know "{source}"? {target}',
            test='{context} Can we know "{source}"? ',
            embed='{context} Can we know "{source}"?',
        )

    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError

@attr.s(auto_attribs=True)
class AGNewsExampleTemplate(ClassificationExampleTemplate):
    choices: list[str] = ["World", "Sports", "Business", "Technology"]
    input_variables: list[str] = ['article', 'label']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='\n{source}\n{target}',
            test='\n{source}\n',
        )

    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError

@attr.s(auto_attribs=True)
class SST2ExampleTemplate(ClassificationExampleTemplate):
    choices: list[str] = ["Negative", "Positive"]
    input_variables: list[str] = ['sentence', 'label']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='\n{source}\n{target}',
            test='\n{source}\n',
        )

    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError
    
@attr.s(auto_attribs=True)
class SarcasmExampleTemplate(ClassificationExampleTemplate):
    choices: list[str] = ["No", "Yes"]
    input_variables: list[str] = ['context', 'label_text']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='\n{source}\n{target}',
            test='\n{source}\n',
        )

    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError


@attr.s(auto_attribs=True)
class IronyExampleTemplate(ClassificationExampleTemplate):
    choices: list[str] = ["No", "Yes"]
    input_variables: list[str] = ['context', 'label_text']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='\n{source}\n{target}',
            test='\n{source}\n',
        )

    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError
    
    

@attr.s(auto_attribs=True)
class QQPExampleTemplate(QNLIExampleTemplate):
    choices: list[str] = ["No", "Yes"]
    input_variables: list[str] = ['question1', 'question2', 'label']

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='{context} Can we say "{source}"? {target}',
            test='{context} Can we say "{source}"? ',
            embed='{context} Can we say "{source}"?',
        )
        
    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()


@attr.s(auto_attribs=True)
class MNLIExampleTemplate(QNLIExampleTemplate):
    choices: list[str] = ["Yes", "Maybe", "No"]
    input_variables: list[str] = ['hypothesis', 'premise', 'label']

    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='{context} Can we say "{source}"? {target}',
            test='{context} Can we say "{source}"? ',
            embed='{context} Can we say "{source}"?',
        )
        
    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()


@attr.s(auto_attribs=True)
class MRPCExampleTemplate(QNLIExampleTemplate):
    choices: list[str] = ["No", "Yes"]
    input_variables: list[str] = ['sentence2', 'sentence1', 'label']
    
    @property
    def templates(self) -> dict[str, str]:
        return dict(
            train='{context} Is this a paraphrased version of "{source}"? {target}',
            test='{context} Is this a paraphrased version of "{source}"? ',
            embed='{context} Is this a paraphrased version of "{source}"?',
        )



@attr.s(auto_attribs=True)
class HellaSwagExampleTemplate(ClassificationExampleTemplate):

    choices: list[str] = ["0","1","2","3"]
    input_variables: list[str] = ['context', 'label']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
                train="Question: {source}\nAnswer: {target}",
                test="Question: {source}\nAnswer: "
            )
    
    
    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError
    

@attr.s(auto_attribs=True)
class CMSQAExampleTemplate(ClassificationExampleTemplate):

    choices: list[str] = ["A","B","C","D", "E"]
    input_variables: list[str] = ['context', 'answerKey']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
                train="{source}\nAnswer: {target}",
                test="{source}\nAnswer: "
            )
    
    
    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError
    
@attr.s(auto_attribs=True)
class MMLUExampleTemplate(ClassificationExampleTemplate):

    choices: list[str] = ["0","1","2","3"]
    input_variables: list[str] = ['context', 'answer']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
                train="{source}\nAnswer: {target}",
                test="{source}\nAnswer: "
            )
    
    
    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError

@attr.s(auto_attribs=True)
class IndMMLUAlgExampleTemplate(ClassificationExampleTemplate):

    choices: list[str] = ["(A)","(B)","(C)","(D)"]
    input_variables: list[str] = ['text', 'label']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
                train="Question: {source}\nAnswer: {target}",
                test="Question: {source}\nAnswer: "
            )
    
    
    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError

@attr.s(auto_attribs=True)
class BANKING77ExampleTemplate(ClassificationExampleTemplate):

    choices: list[str] = ['apple_pay_or_google_pay',
 'supported_cards_and_currencies',
 'pending_cash_withdrawal',
 'terminate_account',
 'receiving_money',
 'top_up_by_bank_transfer_charge',
 'contactless_not_working',
 'card_arrival',
 'fiat_currency_support',
 'pending_card_payment',
 'beneficiary_not_allowed',
 'transfer_fee_charged',
 'transfer_not_received_by_recipient',
 'atm_support',
 'getting_spare_card',
 'country_support',
 'verify_source_of_funds',
 'pending_transfer',
 'edit_personal_details',
 'get_disposable_virtual_card',
 'card_acceptance',
 'card_about_to_expire',
 'compromised_card',
 'reverted_card_payment?',
 'why_verify_identity',
 'pending_top_up',
 'cash_withdrawal_not_recognised',
 'card_payment_wrong_exchange_rate',
 'unable_to_verify_identity',
 'transfer_into_account',
 'visa_or_mastercard',
 'age_limit',
 'declined_cash_withdrawal',
 'change_pin',
 'direct_debit_payment_not_recognised',
 'balance_not_updated_after_cheque_or_cash_deposit',
 'balance_not_updated_after_bank_transfer',
 'card_delivery_estimate',
 'verify_my_identity',
 'exchange_rate',
 'get_physical_card',
 'disposable_card_limits',
 'failed_transfer',
 'verify_top_up',
 'card_swallowed',
 'lost_or_stolen_phone',
 'exchange_via_app',
 'top_up_failed',
 'Refund_not_showing_up',
 'card_not_working',
 'transaction_charged_twice',
 'card_payment_fee_charged',
 'passcode_forgotten',
 'getting_virtual_card',
 'activate_my_card',
 'order_physical_card',
 'top_up_by_card_charge',
 'card_linking',
 'exchange_charge',
 'automatic_top_up',
 'request_refund',
 'pin_blocked',
 'cash_withdrawal_charge',
 'top_up_reverted',
 'cancel_transfer',
 'lost_or_stolen_card',
 'wrong_amount_of_cash_received',
 'topping_up_by_card',
 'virtual_card_not_working',
 'top_up_limits',
 'declined_card_payment',
 'extra_charge_on_statement',
 'wrong_exchange_rate_for_cash_withdrawal',
 'top_up_by_cash_or_cheque',
 'declined_transfer',
 'card_payment_not_recognised',
 'transfer_timing']
    
    input_variables: list[str] = ['context', 'label_text']
    input_label: None = None
    context_label: None = None
    output_label: None = None

    @property
    def templates(self) -> dict[str, str]:
        return dict(
                train="{source}\nAnswer: {target}",
                test="{source}\nAnswer: "
            )
    
    
    def get_embedding_text(self, source, context):
        # return source if not self.embed_context else self.templates['embed'].format(source=source, context=context)
        return self.templates['embed'].format(source=source, context=context)

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def undo_format(self, string):
        raise NotImplementedError
    
# hellaswag_prefix = "Complete the sentence with one of the choices:"
# @attr.s(auto_attribs=True)
# class HellaSwagExampleTemplate(ClassificationExampleTemplate):
#     choices: None = None
#     prompt_format: str = 'QC-A'      # 'Q-A' or 'QC-A'
#     input_variables: list[str] = ['ctx', 'endings', 'label']

#     def get_source(self, **kwargs):
#         return kwargs[self.input_variables[0]].strip()

#     def get_target(self, **kwargs):
#         choices = self.get_choices(**kwargs)
#         answer = choices[int(kwargs[self.input_variables[2]])]
#         return answer

#     def get_choices(self, **kwargs):
#         # print(kwargs)
#         # if kwargs:
#         return [choice.strip() for choice in kwargs[self.input_variables[1]]]
#         # else:
#         #     return []

#     @property
#     def templates(self) -> dict[str, str]:
#         if self.prompt_format == 'QC-A':
#             return dict(
#                 train="Question: {source}\nChoices: {choices}\nAnswer: {target}",
#                 test="Question: {source}\nChoices: {choices}\nAnswer: "
#             )
#         else:
#             return dict(
#                 train="Question: {source}\nAnswer: {target}",
#                 test="Question: {source}\nAnswer: "
#             )

#     def format(self, test=False, embedding=False, **kwargs):
#         source = self.get_source(**kwargs)
#         choices = self.get_choices(**kwargs)
#         target = self.get_target(**kwargs)
#         if embedding: return source
#         template = self.templates['test'] if test else self.templates['train']
#         return template.format(source=source, choices=', '.join(choices), target=target)

#     def parse_output(self, lm_output: str, **kwargs) -> str:
#         return lm_output.strip()

#     def undo_format(self, string):
#         raise NotImplementedError


# cmsqa_prefix = "Answer the questions with one of the given choices:"
# @attr.s(auto_attribs=True)
# class CMSQAExampleTemplate(ClassificationExampleTemplate):
#     choices: None = None
#     prompt_format: str = 'Q-A'      # 'Q-A' or 'QC-A'
#     input_variables: list[str] = ['question', 'choices', 'answerKey']

#     def get_source(self, **kwargs):
#         return kwargs[self.input_variables[0]].strip()

#     def get_target(self, **kwargs):
#         choices = self.get_choices(**kwargs)
#         answer = choices[ord(kwargs[self.input_variables[2]])-ord('A')]
#         return answer

#     def get_choices(self, **kwargs):
#         return [choice.strip() for choice in kwargs[self.input_variables[1]]['text']]

#     @property
#     def templates(self) -> dict[str, str]:
#         if self.prompt_format == 'QC-A':
#             return dict(
#                 train="Question: {source}\nChoices: {choices}\nAnswer: {target}",
#                 test="Question: {source}\nChoices: {choices}\nAnswer: "
#             )
#         else:
#             return dict(
#                 train="Question: {source}\nAnswer: {target}",
#                 test="Question: {source}\nAnswer: "
#             )

#     def format(self, test=False, embedding=False, **kwargs):
#         source = self.get_source(**kwargs)
#         choices = self.get_choices(**kwargs)
#         target = self.get_target(**kwargs)
#         if embedding: return source
#         template = self.templates['test'] if test else self.templates['train']
#         return template.format(source=source, choices=', '.join(choices), target=target)

#     def parse_output(self, lm_output: str, **kwargs) -> str:
#         return lm_output.strip()

#     def undo_format(self, string):
#         raise NotImplementedError

if False:
    @attr.s(auto_attribs=True)
    class ContextualizeGenerationExampleTemplate(GenerationExampleTemplate):
        input_variables: list[str] = ['input', 'context', 'output']
        context_label: str = 'Context'
        version: str = 'with_context'
        embed_context: bool = False

        def get_context(self, **kwargs):
            return kwargs[self.input_variables[1]].strip()

        @property
        def template(self):
            assert self.version == 'with_context', 'ContextualizedClassificationExampleTemplate only works with prompt version="with_context"'
            return common_templates[self.version]

        def get_embedding_text(self, source, context):
            return source if not self.embed_context else f'{context}\n{source}'

        def format(self, test=False, embedding=False, **kwargs):
            source = self.get_source(**kwargs)
            context = self.get_context(**kwargs)
            target = self.get_target(**kwargs)
            if embedding:
                return self.get_embedding_text(source, context)
            template = self.templates['test'] if test else self.templates['train']
            return template.format(
                source=source, context=context, target=target,
                input_label=self.input_label,
                context_label=self.context_label,
                output_label=self.output_label
            )


    @attr.s(auto_attribs=True)
    class ContextualizedClassificationExampleTemplate(ClassificationExampleTemplate):
        input_variables: list[str] = ['text', 'context', 'label']
        context_label: str = 'Context'
        version: str = 'with_context'
        embed_context: bool = False

        def get_context(self, **kwargs):
            return kwargs[self.input_variables[1]].strip()

        @property
        def template(self):
            assert self.version == 'with_context', 'ContextualizedClassificationExampleTemplate only works with prompt version="with_context"'
            return common_templates[self.version]

        def get_embedding_text(self, source, context):
            return source if not self.embed_context else f'{context}\n{source}'

        def format(self, test=False, embedding=False, **kwargs):
            source = self.get_source(**kwargs)
            context = self.get_context(**kwargs)
            target = self.get_target(**kwargs)
            if embedding:
                return self.get_embedding_text(source, context)
            template = self.templates['test'] if test else self.templates['train']
            return template.format(
                source=source, context=context, target=target,
                input_label=self.input_label,
                context_label=self.context_label,
                output_label=self.output_label
            )

