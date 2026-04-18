use crate::{
	components::{Component, color_selector},
	drawing,
	layout::WidgetID,
	parser::{AttribPair, ParserContext, process_component, style::parse_style},
};

pub fn parse_component_color_selector(
	ctx: &mut ParserContext,
	parent_id: WidgetID,
	attribs: &[AttribPair],
	tag_name: &str,
) -> anyhow::Result<WidgetID> {
	let style = parse_style(ctx, attribs, tag_name);
	let color = drawing::Color::new(1.0, 1.0, 1.0, 1.0);

	let (widget, component) = color_selector::construct(
		&mut ctx.get_construct_essentials(parent_id),
		color_selector::Params { color, style },
	)?;

	process_component(ctx, Component(component), widget.id, attribs);

	Ok(widget.id)
}
