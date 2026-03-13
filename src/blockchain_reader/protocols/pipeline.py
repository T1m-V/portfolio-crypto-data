from blockchain_reader.protocols.aave import process_all_aave_tokens
from blockchain_reader.protocols.aura import process_all_aura_tokens
from blockchain_reader.protocols.balancer import process_all_balancer_tokens
from blockchain_reader.protocols.beefy import process_all_beefy_tokens
from blockchain_reader.protocols.composer import compose_base_ingredients
from blockchain_reader.protocols.curve import process_all_curve_tokens


def run_protocol_pipeline(
    chain: str,
    protocols: list[str] | None = None,
    start_date: str | None = None,
) -> None:
    selected = set(protocols or ["beefy", "balancer", "aave", "aura", "curve"])

    if "beefy" in selected:
        process_all_beefy_tokens(chain=chain, start_date=start_date)
    if "balancer" in selected:
        process_all_balancer_tokens(chain=chain, start_date=start_date)
    if "aura" in selected:
        process_all_aura_tokens(chain=chain, start_date=start_date)
    if "curve" in selected:
        process_all_curve_tokens(chain=chain, start_date=start_date)
    if "aave" in selected:
        process_all_aave_tokens(chain=chain, start_date=start_date)

    compose_base_ingredients(chain=chain)


if __name__ == "__main__":
    run_protocol_pipeline(
        chain="arbitrum",
        protocols=["beefy", "balancer", "aave", "aura", "curve"],
    )
